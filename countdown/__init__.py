import sys

import jax
import jax.numpy as jnp
import numpy as np
import optax
from anndata import AnnData
from flax import nnx
from jax._src.dtypes import JAXType
from jax.experimental.sparse import BCSR, bcsr_extract
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from tqdm import tqdm


def as_dense_f32(X: csr_matrix | np.ndarray) -> np.ndarray:
    if isinstance(X, csr_matrix):
        return X.todense().astype(np.float32)
    else:
        return X.astype(np.float32)


class CSRMatrixRowSampler:
    """Samples random batches of rows from a CSR matrix, yielding dense JAX arrays."""

    def __init__(self, X: csr_matrix, batch_size: int):
        m, n = X.shape
        self.X = X.astype(np.float32)
        self.idx = np.arange(m)
        self.chunk = np.zeros((batch_size, n), dtype=np.float32)
        self.batch_size = batch_size
        self.m = m
        self.n = n

    def __iter__(self):
        np.random.shuffle(self.idx)
        for fr in range(0, len(self.idx), self.batch_size):
            to = min(fr + self.batch_size, self.m)
            batch_size = to - fr
            batch_indices = self.idx[fr:to]
            batch_indices.sort()

            if batch_size == self.batch_size:
                self.X[batch_indices, :].todense(out=self.chunk)
                yield jnp.array(self.chunk)
            else:
                # Final partial batch - create appropriately sized array
                partial_chunk = np.zeros((batch_size, self.n), dtype=np.float32)
                self.X[batch_indices, :].todense(out=partial_chunk)
                yield jnp.array(partial_chunk)


class PaddedBCSRSampler:
    """
    Samples batches of rows from a CSR matrix as padded BCSR arrays.

    Precomputes padded batch data once at initialization (stored in CPU memory),
    then converts to BCSR on-demand during iteration. All batches are padded to
    uniform nse (number of stored elements) to avoid JIT recompilation.
    """

    def __init__(self, X: csr_matrix, batch_size: int):
        m, n = X.shape
        X = X.astype(np.float32)
        self.n = n

        # Shuffle once at initialization
        idx = np.arange(m)
        np.random.shuffle(idx)

        # First pass: determine max nse across all batches
        max_nse = 0
        batch_indices_list = []
        for fr in range(0, m, batch_size):
            to = min(fr + batch_size, m)
            batch_idx = idx[fr:to].copy()
            batch_idx.sort()
            batch_indices_list.append(batch_idx)
            sliced = X[batch_idx, :]
            max_nse = max(max_nse, sliced.nnz)

        # Second pass: precompute padded numpy arrays for each batch
        self._batches: list[tuple[np.ndarray, np.ndarray, np.ndarray, int]] = []
        for batch_idx in batch_indices_list:
            sliced = X[batch_idx, :]
            batch_m = sliced.shape[0]
            current_nse = sliced.nnz
            pad_size = max_nse - current_nse

            if pad_size > 0:
                data = np.concatenate(
                    [sliced.data, np.zeros(pad_size, dtype=np.float32)]
                )
                indices = np.concatenate(
                    [sliced.indices, np.zeros(pad_size, dtype=np.int32)]
                )
                indptr = sliced.indptr.copy()
                indptr[-1] = max_nse
            else:
                data = sliced.data.copy()
                indices = sliced.indices.copy()
                indptr = sliced.indptr.copy()

            self._batches.append((data, indices, indptr, batch_m))

    def __iter__(self):
        for data, indices, indptr, batch_m in self._batches:
            yield BCSR(
                (
                    jnp.array(data),
                    jnp.array(indices, dtype=jnp.int32),
                    jnp.array(indptr, dtype=jnp.int32),
                ),
                shape=(batch_m, self.n),
            )


class DenseMatrixRowSampler:
    def __init__(self, X: np.ndarray, batch_size: int):
        m, n = X.shape
        self.X = X
        self.idx = np.arange(m)
        self.chunk = np.zeros((batch_size, n), dtype=np.float32)
        self.batch_size = batch_size
        self.m = m
        self.n = n

    def __iter__(self):
        np.random.shuffle(self.idx)
        for fr in range(0, len(self.idx), self.batch_size):
            to = min(fr + self.batch_size, self.m)
            batch_size = to - fr
            if batch_size == self.batch_size:
                self.chunk[:] = self.X[self.idx[fr:to], :]
                yield jnp.array(self.chunk)
            else:
                # Final partial batch
                partial_chunk = self.X[self.idx[fr:to], :].astype(np.float32)
                yield jnp.array(partial_chunk)


class SimpleEncoder(nnx.Module):
    """
    Simple single-layer encoder: X -> linear -> softplus -> U

    Architecture:
    - Single linear projection from input to factors
    - Separate linear layer for scale prediction
    - Fast, low correlation, good baseline

    Best for: Quick results, interpretability, low metagene correlation
    """

    def __init__(self, n: int, k: int, *, rngs: nnx.Rngs):
        self.weights = nnx.Param(
            nnx.initializers.lecun_normal()(rngs.params(), (n, k), jnp.float32)
        )
        self.bias = nnx.Param(jnp.zeros(k))

    def __call__(self, X: jax.Array | BCSR):
        u = nnx.softplus(X @ self.weights[...] + self.bias[...])
        return u


class DeepSoftplusEncoder(nnx.Module):
    """
    Two-layer encoder with softplus activations: X -> hidden -> output

    Architecture:
    - Input -> hidden_dim (softplus) -> k (softplus)
    - Softplus activations work well with count data
    - Scale predicted directly from input

    Best for: Maximum likelihood with regularization
    Recommended with: metagene_reg_type="correlation", metagene_reg_strength=0.01
    """

    def __init__(self, n: int, k: int, hidden_dim: int, *, rngs: nnx.Rngs):
        self.weights1 = nnx.Param(
            nnx.initializers.lecun_normal()(rngs.params(), (n, hidden_dim), jnp.float32)
        )
        self.bias1 = nnx.Param(jnp.zeros(hidden_dim))

        self.lyr2 = nnx.Linear(hidden_dim, k, rngs=rngs)

    def __call__(self, X: jax.Array | BCSR):
        h = X @ self.weights1[...] + self.bias1[...]
        h = nnx.softplus(h)
        h = self.lyr2(h)
        u = nnx.softplus(h)

        return u


class DeepCompactEncoder(nnx.Module):
    """
    Compact two-layer encoder: X -> k-dim hidden -> k output

    Architecture:
    - Input -> k (softplus) -> k (softplus)
    - Reduced capacity vs DeepSoftplusEncoder (k vs hidden_dim)
    - Scale predicted directly from input

    Best for: Good likelihood with lower capacity, works well with regularization
    """

    def __init__(self, n: int, k: int, hidden_dim: int, *, rngs: nnx.Rngs):
        # Use k as hidden dim instead of hidden_dim to match SimpleEncoder capacity
        self.weights1 = nnx.Param(
            nnx.initializers.lecun_normal()(rngs.params(), (n, k), jnp.float32)
        )
        self.bias1 = nnx.Param(jnp.zeros(k))

        self.lyr2 = nnx.Linear(k, k, rngs=rngs)

    def __call__(self, X: jax.Array | BCSR):
        h = X @ self.weights1[...] + self.bias1[...]
        h = nnx.softplus(h)
        h = self.lyr2(h)
        u = nnx.softplus(h)

        return u


class BoundedAuxiliaryEncoder(nnx.Module):
    """
    Direct + bounded auxiliary path encoder

    Architecture:
    - Primary: X -> linear
    - Auxiliary: X -> linear -> tanh (bounded to [-1, 1], scaled 0.1)
    - Combined: softplus(primary + auxiliary)

    This architectural approach maintains low correlation without regularization
    by limiting auxiliary path influence through tanh bounding.

    Best for: Good likelihood + low correlation without needing regularization
    """

    def __init__(self, n: int, k: int, hidden_dim: int, *, rngs: nnx.Rngs):
        # Primary path: direct like SimpleEncoder
        self.weights_direct = nnx.Param(
            nnx.initializers.lecun_normal()(rngs.params(), (n, k), jnp.float32)
        )
        self.bias_direct = nnx.Param(jnp.zeros(k))

        # Auxiliary path for refinement (small weight 0.1)
        self.weights_aux = nnx.Param(
            nnx.initializers.lecun_normal()(rngs.params(), (n, k), jnp.float32) * 0.1
        )
        self.bias_aux = nnx.Param(jnp.zeros(k))

    def __call__(self, X: jax.Array | BCSR):
        # Main direct path
        u_direct = X @ self.weights_direct[...] + self.bias_direct[...]

        # Auxiliary refinement
        u_aux = X @ self.weights_aux[...] + self.bias_aux[...]
        u_aux = jnp.tanh(u_aux)  # bounded correction

        # Combine and apply softplus
        u = nnx.softplus(u_direct + u_aux)

        return u


class ScaleEncoder(nnx.Module):
    """
    Simple MLP encoder for predicting cell-specific scale factors.

    Architecture:
    - Input (n genes) -> hidden layer (16 units) -> softplus -> output (1 scale)
    - Small hidden layer (16) for stability with high-dimensional sparse input
    - Predicts log_scale which is clipped and exponentiated

    Trained separately from main encoder to provide stable scale predictions.
    """

    def __init__(self, n: int, *, rngs: nnx.Rngs):
        # Small hidden layer for stability
        hidden_dim = 32

        self.weights1 = nnx.Param(
            1e-3 * jax.random.normal(rngs.params(), (n, hidden_dim), dtype=jnp.float32)
        )
        self.bias1 = nnx.Param(jnp.zeros(hidden_dim))

        self.weights2 = nnx.Param(
            1e-3 * jax.random.normal(rngs.params(), (hidden_dim, 1), dtype=jnp.float32)
        )
        self.bias2 = nnx.Param(jnp.zeros(1))

    def __call__(self, X: jax.Array | BCSR):
        # Hidden layer with softplus activation
        h = X @ self.weights1[...] + self.bias1[...]
        h = nnx.softplus(h)

        # Output layer (log_scale)
        log_scale = h @ self.weights2[...] + self.bias2[...]

        return log_scale


class NMF(nnx.Module):
    def __init__(
        self,
        n: int,
        k: int,
        hidden_dim: int,
        *,
        rngs: nnx.Rngs,
        r_prior_alpha: float = 2.0,
        r_prior_beta: float = 2.0,
        scale_prior_sigma: float = 0.5,
        encoder_version: str = "simple",
        metagene_reg_type: str = "none",
        metagene_reg_strength: float = 0.01,
        gene_scale_factors: bool = False,
        init_method: str = "normal",
        v_init: jax.Array | None = None,
    ):
        key = rngs.params()

        # Select encoder based on version
        if encoder_version == "simple":
            self.encoder = SimpleEncoder(n, k, rngs=rngs)
        elif encoder_version == "deep_softplus":
            self.encoder = DeepSoftplusEncoder(n, k, hidden_dim=hidden_dim, rngs=rngs)
        elif encoder_version == "deep_compact":
            self.encoder = DeepCompactEncoder(n, k, hidden_dim=hidden_dim, rngs=rngs)
        elif encoder_version == "bounded_auxiliary":
            self.encoder = BoundedAuxiliaryEncoder(
                n, k, hidden_dim=hidden_dim, rngs=rngs
            )
        else:
            raise ValueError(
                f"Unknown encoder_version: {encoder_version}. "
                f"Must be one of: 'simple', 'deep_softplus', 'deep_compact', 'bounded_auxiliary'"
            )

        if v_init is not None:
            self.v = nnx.Param(v_init)
        elif init_method == "normal":
            self.v = nnx.Param(jax.random.normal(key, (k, n)) / jnp.sqrt(n))
        elif init_method == "uniform":
            self.v = nnx.Param(
                jax.random.uniform(key, (k, n), minval=-1.0, maxval=1.0) / jnp.sqrt(n)
            )
        elif init_method == "kaiming":
            self.v = nnx.Param(nnx.initializers.he_normal()(key, (k, n), jnp.float32))
        elif init_method == "xavier":
            self.v = nnx.Param(
                nnx.initializers.xavier_normal()(key, (k, n), jnp.float32)
            )
        elif init_method == "orthogonal":
            self.v = nnx.Param(nnx.initializers.orthogonal()(key, (k, n), jnp.float32))
        else:
            raise ValueError(f"Unknown init_method: {init_method}")

        self.log_r = nnx.Param(jnp.full(n, 1e-1))

        # Separate scale encoder - simple MLP for predicting cell-specific scales
        self.scale_encoder = ScaleEncoder(n, rngs=rngs)

        # Store prior hyperparameters (not trainable)
        self.r_prior_alpha = r_prior_alpha
        self.r_prior_beta = r_prior_beta
        self.scale_prior_sigma = scale_prior_sigma
        self.metagene_reg_type = metagene_reg_type
        self.metagene_reg_strength = metagene_reg_strength
        self.gene_scale_factors = gene_scale_factors

    # X: [batch_size, n]
    def __call__(self, X: jax.Array | BCSR) -> jax.Array:
        u = self.encoder(X)  # [batch_size, k]

        # Predict log_scale using separate scale encoder
        log_scale = self.scale_encoder(X)  # [batch_size, 1]

        # Clip log_scale to prevent overflow in exp()
        # Allows scales from exp(-10) ≈ 0.000045 to exp(10) ≈ 22026
        log_scale = jnp.clip(log_scale, -10.0, 10.0)

        scale = jnp.exp(log_scale)  # [batch_size, 1]

        # jax.debug.print("mean scale: {}", jnp.mean(scale))
        # jax.debug.print("scale: [{}, {}]", jnp.min(scale), jnp.max(scale))

        lambda_base = u @ self.v_scaled()  # [batch_size, n]
        # Scale each cell's predictions by its scale factor

        if self.gene_scale_factors:
            lambda_scaled = lambda_base * scale  # [batch_size, n]
        else:
            lambda_scaled = lambda_base

        return lambda_scaled, u, log_scale

    def v_norm(self) -> jax.Array:
        return nnx.softmax(self.v[...], axis=1)

    def v_scaled(self) -> jax.Array:
        return self.v_norm()

    def r(self) -> jax.Array:
        return jnp.exp(self.log_r[...])

    def metagene_regularization(self, u: jax.Array) -> jax.Array:
        """
        Compute regularization term to encourage diverse/orthogonal metagenes.

        Args:
            u: [batch_size, k] metagene usage matrix from current batch

        Returns penalty term (to be minimized).
        """
        if self.metagene_reg_type == "none":
            return 0.0

        # Get normalized V matrix (k x n), each row is a metagene
        v_norm = self.v_norm()  # [k, n]

        if self.metagene_reg_type == "correlation":
            # Penalize correlation between metagene rows
            # Compute correlation matrix: corr[i,j] = correlation between metagene i and j
            # First center each row
            v_centered = v_norm - jnp.mean(v_norm, axis=1, keepdims=True)
            # Compute row norms
            row_norms = jnp.sqrt(jnp.sum(v_centered**2, axis=1, keepdims=True))
            # Normalize
            v_normalized = v_centered / (row_norms + 1e-8)
            # Compute correlation matrix
            corr_matrix = v_normalized @ v_normalized.T  # [k, k]
            # Penalize off-diagonal elements (we want them near 0)
            # Extract off-diagonal using mask
            k = corr_matrix.shape[0]
            mask = 1.0 - jnp.eye(k)
            off_diag_corr = corr_matrix * mask
            penalty = jnp.sum(off_diag_corr**2)
            return self.metagene_reg_strength * penalty

        elif self.metagene_reg_type == "orthogonal":
            # Penalize V @ V^T being far from identity
            # V is [k, n], so V @ V^T is [k, k]
            gram = v_norm @ v_norm.T  # [k, k]
            k = gram.shape[0]
            identity = jnp.eye(k)
            penalty = jnp.sum((gram - identity) ** 2)
            return self.metagene_reg_strength * penalty

        elif self.metagene_reg_type == "cosine":
            # Penalize cosine similarity between metagene pairs
            # Normalize rows to unit length
            row_norms = jnp.sqrt(jnp.sum(v_norm**2, axis=1, keepdims=True))
            v_unit = v_norm / (row_norms + 1e-8)
            # Compute cosine similarity matrix
            cosine_sim = v_unit @ v_unit.T  # [k, k]
            # Penalize off-diagonal (want low similarity)
            k = cosine_sim.shape[0]
            mask = 1.0 - jnp.eye(k)
            off_diag_sim = cosine_sim * mask
            penalty = jnp.sum(off_diag_sim**2)
            return self.metagene_reg_strength * penalty

        elif self.metagene_reg_type == "entropy":
            # Encourage even usage of metagenes across cells
            # Compute mean usage per metagene across this batch
            u_mean = jnp.mean(u, axis=0)  # [k]
            # Normalize to sum to 1
            u_prob = u_mean / (jnp.sum(u_mean) + 1e-8)
            # Compute entropy (higher = more even usage)
            entropy = -jnp.sum(u_prob * jnp.log(u_prob + 1e-8))
            # Penalize low entropy (we want to maximize entropy, so minimize negative)
            max_entropy = jnp.log(float(u.shape[1]))  # log(k)
            penalty = max_entropy - entropy
            return self.metagene_reg_strength * penalty

        elif self.metagene_reg_type == "usage":
            # Penalize unused metagenes
            # Count which metagenes are active (>threshold) in this batch
            threshold = 0.1
            # Use sigmoid for differentiable approximation of indicator function (u > threshold)
            # steepness=100 makes transition sharp enough to avoid false positives for zero inputs
            active = jnp.mean(
                jax.nn.sigmoid(100.0 * (u - threshold)), axis=0
            )  # [k] - fraction of cells using each metagene
            # Penalize metagenes with low usage
            penalty = jnp.sum(jnp.maximum(0.01 - active, 0.0))  # Penalize if <1% usage
            return self.metagene_reg_strength * penalty

        elif self.metagene_reg_type == "weighted_correlation":
            # Weight correlation by gene variance to focus on expressed genes
            # This addresses the "useless orthogonal metagenes" problem

            # Compute gene weights from V (high variance in V = important gene)
            gene_variance = jnp.var(v_norm, axis=0)  # [n]
            gene_weights = gene_variance / (jnp.sum(gene_variance) + 1e-8)

            # Weight each gene's contribution to correlation
            v_weighted = v_norm * jnp.sqrt(gene_weights)  # Weight before correlation

            # Compute correlation on weighted V
            v_centered = v_weighted - jnp.mean(v_weighted, axis=1, keepdims=True)
            row_norms = jnp.sqrt(jnp.sum(v_centered**2, axis=1, keepdims=True))
            v_normalized = v_centered / (row_norms + 1e-8)
            corr_matrix = v_normalized @ v_normalized.T

            # Penalize off-diagonal
            k = corr_matrix.shape[0]
            mask = 1.0 - jnp.eye(k)
            penalty = jnp.sum((corr_matrix * mask) ** 2)
            return self.metagene_reg_strength * penalty

        elif self.metagene_reg_type == "combined":
            # Combine weighted correlation + entropy regularization
            # This should give decorrelated metagenes that are actually used

            # 1. Weighted correlation (focus on expressed genes)
            gene_variance = jnp.var(v_norm, axis=0)
            gene_weights = gene_variance / (jnp.sum(gene_variance) + 1e-8)
            v_weighted = v_norm * jnp.sqrt(gene_weights)
            v_centered = v_weighted - jnp.mean(v_weighted, axis=1, keepdims=True)
            row_norms = jnp.sqrt(jnp.sum(v_centered**2, axis=1, keepdims=True))
            v_normalized = v_centered / (row_norms + 1e-8)
            corr_matrix = v_normalized @ v_normalized.T
            k = corr_matrix.shape[0]
            mask = 1.0 - jnp.eye(k)
            corr_penalty = jnp.sum((corr_matrix * mask) ** 2)

            # 2. Entropy (encourage even usage)
            u_mean = jnp.mean(u, axis=0)
            u_prob = u_mean / (jnp.sum(u_mean) + 1e-8)
            entropy = -jnp.sum(u_prob * jnp.log(u_prob + 1e-8))
            max_entropy = jnp.log(float(u.shape[1]))
            entropy_penalty = max_entropy - entropy

            # Combine with equal weight
            return self.metagene_reg_strength * (corr_penalty + entropy_penalty)

        elif self.metagene_reg_type == "balanced":
            # Like combined, but weighs entropy significantly higher to force usage

            # 1. Weighted correlation
            gene_variance = jnp.var(v_norm, axis=0)
            gene_weights = gene_variance / (jnp.sum(gene_variance) + 1e-8)
            v_weighted = v_norm * jnp.sqrt(gene_weights)
            v_centered = v_weighted - jnp.mean(v_weighted, axis=1, keepdims=True)
            row_norms = jnp.sqrt(jnp.sum(v_centered**2, axis=1, keepdims=True))
            v_normalized = v_centered / (row_norms + 1e-8)
            corr_matrix = v_normalized @ v_normalized.T
            k = corr_matrix.shape[0]
            mask = 1.0 - jnp.eye(k)
            corr_penalty = jnp.sum((corr_matrix * mask) ** 2)

            # 2. Entropy
            u_mean = jnp.mean(u, axis=0)
            u_prob = u_mean / (jnp.sum(u_mean) + 1e-8)
            entropy = -jnp.sum(u_prob * jnp.log(u_prob + 1e-8))
            max_entropy = jnp.log(float(u.shape[1]))
            entropy_penalty = max_entropy - entropy

            # Higher weight on entropy (1,000,000x)
            return self.metagene_reg_strength * (
                corr_penalty + 1000000.0 * entropy_penalty
            )

        else:
            return 0.0

    def log_prior(self, u: jax.Array, log_scale: jax.Array) -> jax.Array:
        """
        Compute log prior for all parameters.

        Currently implements:
        - Gamma(alpha, beta) prior on dispersion parameters r
        - Normal(0, sigma) prior on scale_logit (cell-specific scale factors)
        - Optional metagene regularization (correlation, orthogonal, or cosine)

        Args:
            scale_logit: [batch_size] array of scale logit values for current batch

        Returns negative log prior (to be minimized).
        """
        # Gamma prior on r
        r = self.r()
        alpha = self.r_prior_alpha
        beta = self.r_prior_beta

        log_prior_r = (
            alpha * jnp.log(beta)
            - jax.scipy.special.gammaln(alpha)
            + (alpha - 1) * jnp.log(r)
            - beta * r
        )

        neg_log_prior = -jnp.sum(log_prior_r)

        # Normal(0, sigma) prior on scale_logit
        # p(scale_logit) = (1 / sqrt(2*pi*sigma^2)) * exp(-scale_logit^2 / (2*sigma^2))
        # log p(scale_logit) = -0.5*log(2*pi) - log(sigma) - scale_logit^2 / (2*sigma^2)
        sigma = self.scale_prior_sigma
        log_prior_scale = (
            -0.5 * jnp.log(2 * jnp.pi) - jnp.log(sigma) - log_scale**2 / (2 * sigma**2)
        )

        neg_log_prior += -jnp.sum(log_prior_scale)

        # Add metagene regularization (pass u for usage-based penalties)
        # Now supports entropy, usage, weighted_correlation, and combined modes
        neg_log_prior += self.metagene_regularization(u)

        # c = 100.0
        # neg_log_prior += -c * jnp.sum(jax.scipy.special.entr(usage))

        return neg_log_prior


def neg_poisson_logprob_dense(model: NMF, X: jax.Array, constant_terms: bool = False):
    """Negative log probability for dense input."""
    λ, u, log_scale = model(X)
    lp = (X * jnp.log(jnp.clip(λ, 1e-8))).sum() - jnp.sum(λ)

    # constant wrt to parameters
    if constant_terms:
        lp -= jnp.sum(jax.scipy.special.gammaln(X + 1))

    neg_log_likelihood = -lp

    # Add negative log prior (for MAP estimation)
    neg_log_posterior = neg_log_likelihood + model.log_prior(u, log_scale)

    return neg_log_posterior


def neg_poisson_logprob_sparse(model: NMF, X: BCSR, constant_terms: bool = False):
    """Negative log probability for sparse BCSR input."""
    λ, u, log_scale = model(X)
    # Extract λ values only at non-zero positions of X
    lp = (
        X.data * jnp.log(jnp.clip(bcsr_extract(X.indices, X.indptr, λ), 1e-8))
    ).sum() - jnp.sum(λ)

    # constant wrt to parameters
    if constant_terms:
        lp -= jnp.sum(jax.scipy.special.gammaln(X.data + 1))

    neg_log_likelihood = -lp

    # Add negative log prior (for MAP estimation)
    neg_log_posterior = neg_log_likelihood + model.log_prior(u, log_scale)

    return neg_log_posterior


def neg_nb_logprob_dense(model: NMF, X: jax.Array, constant_terms: bool = False):
    λ, u, log_scale = model(X)  # [ncells, ngenes], [ncells]
    r = jnp.expand_dims(model.r(), 0)  # [1, ngenes]
    log_r = jnp.expand_dims(model.log_r[...], 0)  # [1, ngenes]
    log_λr = jnp.log(λ + r)  # [ncells, ngenes]
    log_λ = jnp.log(λ)
    gammaln_r = jax.scipy.special.gammaln(r)  # [1, ngenes]

    ncells = X.shape[0]

    lp = 0.0
    lp += -jnp.sum(ncells * gammaln_r)

    lp += jnp.sum(ncells * r * log_r)
    lp -= jnp.sum(r * jnp.sum(log_λr, axis=0, keepdims=True))
    lp += jnp.sum(X * (log_λ - log_λr))
    lp += jnp.sum(jax.scipy.special.gammaln(X + r))

    # constant wrt to parameters
    if constant_terms:
        lp -= jnp.sum(jax.scipy.special.gammaln(X + 1))

    neg_log_likelihood = -lp

    # Add negative log prior (for MAP estimation)
    neg_log_posterior = neg_log_likelihood + model.log_prior(u, log_scale)

    return neg_log_posterior


def neg_nb_logprob_sparse(model: NMF, X: BCSR, constant_terms: bool = False):
    λ, u, log_scale = model(X)  # [ncells, ngenes], [ncells]
    r = jnp.expand_dims(model.r(), 0)  # [1, ngenes]
    log_r = jnp.expand_dims(model.log_r[...], 0)  # [1, ngenes]
    log_λr = jnp.log(λ + r)  # [ncells, ngenes]
    log_λ = jnp.log(λ)
    gammaln_r = jax.scipy.special.gammaln(r)  # [1, ngenes]

    ncells = X.shape[0]

    lp = 0.0
    lp += -jnp.sum(ncells * gammaln_r)

    lp += jnp.sum(ncells * r * log_r)
    lp -= jnp.sum(r * jnp.sum(log_λr, axis=0, keepdims=True))

    lp += jnp.sum(X.data * bcsr_extract(X.indices, X.indptr, log_λ - log_λr))

    # This is the naive implementation of this term, which requires
    # densifying X and redundantly computing gammaln across many identical values.
    # lp += jnp.sum(jax.scipy.special.gammaln(X.todense() + r))

    # Alternative, we can just do this, then subtract out the terms we want, etc.
    lp += jnp.sum(ncells * gammaln_r)

    # subtract out the log(gamma(r)) values where X is nonzero.
    lp -= jnp.sum(gammaln_r[0, X.indices])

    # compute just the log(gamma(r + x)) values where X is nonzero
    lp += jnp.sum(jax.scipy.special.gammaln(r[0, X.indices] + X.data))

    # constant wrt to parameters
    if constant_terms:
        lp -= jnp.sum(jax.scipy.special.gammaln(X.data + 1))

    neg_log_likelihood = -lp

    # Add negative log prior (for MAP estimation)
    neg_log_posterior = neg_log_likelihood + model.log_prior(u, log_scale)

    return neg_log_posterior


def create_train_step_dense(likelihood: str):
    """Create a JIT-compiled training step for dense input with specified likelihood."""
    if likelihood == "nb":
        loss_fn = neg_nb_logprob_dense
    elif likelihood == "poisson":
        loss_fn = neg_poisson_logprob_dense
    else:
        raise ValueError(
            f"Unknown likelihood: {likelihood}. Must be 'nb' or 'poisson'."
        )

    @nnx.jit
    def train_step(model: NMF, optimizer: nnx.Optimizer, X: jax.Array):
        """Fused forward, backward, and optimizer update for dense input."""
        loss, grads = nnx.value_and_grad(loss_fn)(model, X)
        optimizer.update(model, grads)
        return loss

    return train_step


def create_train_step_sparse(likelihood: str):
    """Create a JIT-compiled training step for sparse BCSR input with specified likelihood."""
    if likelihood == "nb":
        loss_fn = neg_nb_logprob_sparse
    elif likelihood == "poisson":
        loss_fn = neg_poisson_logprob_sparse
    else:
        raise ValueError(
            f"Unknown likelihood: {likelihood}. Must be 'nb' or 'poisson'."
        )

    @nnx.jit
    def train_step(model: NMF, optimizer: nnx.Optimizer, X: BCSR):
        """Fused forward, backward, and optimizer update for sparse BCSR input."""
        loss, grads = nnx.value_and_grad(loss_fn)(model, X)
        optimizer.update(model, grads)
        return loss

    return train_step


def nmf(
    adata: AnnData,
    k: int = 64,
    batch_size: int | None = 4096,
    hidden_dim=512,
    lr=1e-2,
    max_epochs: int = 2000,
    patience: int = 40,
    min_delta: float = 1e-5,
    sparse: bool = True,
    likelihood: str = "nb",
    r_prior_alpha: float = 2.0,
    r_prior_beta: float = 2.0,
    scale_prior_sigma: float = 0.1,
    encoder_version: str = "bounded_auxiliary",
    metagene_reg_type: str = "none",
    metagene_reg_strength: float = 0.01,
    gene_scale_factors: bool = True,
    init_method: str = "nndsvd",
    init_ncells: int = 10000,
    optimizer_name: str = "adam",
    quiet: bool = False,
    filter_min_prop: float = 1e-5,
    filter_min_delta: float = 1.0,
):
    """
    Perform Non-negative Matrix Factorization (NMF) on genomic count data.

    This function applies a neural network-based NMF to decompose the count matrix
    stored in an AnnData object into lower-dimensional representations. The method
    uses an encoder-decoder architecture with early stopping for convergence.

    Parameters
    ----------
    adata : AnnData
        Annotated data object containing the count matrix in adata.X.
        The matrix should have shape (n_observations, n_features).
    k : int, default=64
        Number of latent factors/components for the NMF decomposition.
        This determines the dimensionality of the reduced representation.
    batch_size : int or None, default=4096
        Size of batches for mini-batch training. If None, uses the full
        dataset (batch_size = n_observations) for training.
    hidden_dim : int, default=512
        Number of hidden units in the encoder neural network.
    lr : float, default=1e-2
        Learning rate for the Adam optimizer.
    max_epochs : int, default=2000
        Maximum number of training epochs.
    patience : int, default=40
        Number of epochs to wait for improvement before early stopping.
    min_delta : float, default=1e-5
        Minimum change in log-probability to be considered an improvement
        for early stopping.
    sparse : bool, default=True
        If True, use sparse BCSR batches (memory efficient, allows larger batches).
        If False, use dense batches (may be faster for smaller matrices).
    likelihood : str, default="nb"
        Likelihood function to use for the NMF model. Either "nb" or "poisson".
    r_prior_alpha : float, default=2.0
        Shape parameter (alpha) for the Gamma prior on dispersion parameters r.
        For Gamma(alpha, beta), the mean is alpha/beta and mode is (alpha-1)/beta.
        Larger values make the prior more concentrated around the mean.
    r_prior_beta : float, default=2.0
        Rate parameter (beta) for the Gamma prior on dispersion parameters r.
        For Gamma(alpha, beta), the mean is alpha/beta and mode is (alpha-1)/beta.
        Larger values push the prior towards smaller r values.
    scale_prior_sigma : float, default=0.1
        Standard deviation for the Normal(0, sigma) prior on log scale values.
        Cell-specific scale factors are learned by the encoder using exp transformation.
        Smaller sigma values enforce stronger regularization towards uniform scaling across cells.
    init_ncells : int, default=100000
        Maximum number of cells to use for NNDSVD initialization. If the number of
        cells in the dataset exceeds this value, a random subset is used to compute
        the initialization, which speeds up startup time for large datasets.

    Returns
    -------
    None
        The function modifies the input AnnData object in-place.

    Notes
    -----
    Results are stored in the AnnData object as:

    - `adata.obsm["X_nmf"]` : ndarray of shape (n_observations, k)
        The full low-dimensional NMF representation of the observations.
    - `adata.obsm["X_nmf_filtered"]` : ndarray of shape (n_observations, k_filtered)
        The filtered NMF representation, removing dimensions with low expression.
    - `adata.varm["V_nmf"]` : ndarray of shape (n_features, k)
        The full factor matrix.
    - `adata.varm["V_nmf_filtered"]` : ndarray of shape (n_features, k_filtered)
        The filtered factor matrix.

    The method supports both sparse (CSR) and dense numpy arrays as input
    and automatically handles batching for memory efficiency with large datasets.
    Training uses early stopping based on log-probability improvement to
    prevent overfitting.

    Examples
    --------
    >>> import anndata as ad
    >>> import numpy as np
    >>> from countdown import nmf
    >>>
    >>> # Create example count data
    >>> X = np.random.poisson(5, size=(1000, 2000))
    >>> adata = ad.AnnData(X)
    >>>
    >>> # Apply NMF with 32 components
    >>> nmf(adata, k=32)
    >>>
    >>> # Access the results
    >>> print(adata.obsm["X_nmf"].shape)  # (1000, 32)
    """
    m, n = adata.shape

    if batch_size is None:
        batch_size = m

    v_init = None
    if init_method == "nndsvd":
        if not quiet:
            print("Computing NNDSVD initialization...")

        # Subsample if dataset is too large
        X_init = adata.X
        if m > init_ncells:
            if not quiet:
                print(f"Subsampling {init_ncells} cells for initialization...")
            # Use fixed seed for reproducibility of initialization subset
            rng = np.random.default_rng(42)
            indices = rng.choice(m, init_ncells, replace=False)
            indices.sort()
            X_init = X_init[indices]

        # Compute SVD
        U, S, Vt = svds(X_init, k=k)
        # Sort by singular values descending
        idx = np.argsort(S)[::-1]
        S = S[idx]
        U = U[:, idx]
        Vt = Vt[idx, :]

        # NNDSVD initialization for H (which becomes V in our model)
        H = np.zeros((k, n))
        for j in range(k):
            x = U[:, j]
            y = Vt[j, :]
            xp = np.maximum(x, 0)
            xn = np.abs(np.minimum(x, 0))
            yp = np.maximum(y, 0)
            yn = np.abs(np.minimum(y, 0))
            xpn = np.linalg.norm(xp)
            xnn = np.linalg.norm(xn)
            ypn = np.linalg.norm(yp)
            ynn = np.linalg.norm(yn)
            mp = xpn * ypn
            mn = xnn * ynn
            if mp > mn:
                v = yp * xpn * np.sqrt(S[j])
            else:
                v = yn * xnn * np.sqrt(S[j])
            H[j, :] = v

        # Add small epsilon to avoid log(0) and zeros
        H = H + 1e-6
        # Normalize rows to sum to 1 to match softmax expectation
        H_norm = H / H.sum(axis=1, keepdims=True)
        # Inverse softmax (approximate) - just use log probabilities
        v_init = jnp.array(np.log(H_norm))

    rngs = nnx.Rngs(0)
    model = NMF(
        n,
        k,
        hidden_dim,
        rngs=rngs,
        r_prior_alpha=r_prior_alpha,
        r_prior_beta=r_prior_beta,
        scale_prior_sigma=scale_prior_sigma,
        encoder_version=encoder_version,
        metagene_reg_type=metagene_reg_type,
        metagene_reg_strength=metagene_reg_strength,
        gene_scale_factors=gene_scale_factors,
        init_method=init_method,
        v_init=v_init,
    )

    if optimizer_name == "adam":
        opt = optax.adam(lr)
    elif optimizer_name == "adamw":
        opt = optax.adamw(lr)
    elif optimizer_name == "sgd":
        opt = optax.sgd(lr, momentum=0.9)
    elif optimizer_name == "rmsprop":
        opt = optax.rmsprop(lr)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    optimizer = nnx.Optimizer(model, opt, wrt=nnx.Param)
    metrics = nnx.MultiMetric(neg_logprob=nnx.metrics.Average("neg_logprob"))

    X = adata.X

    if sparse:
        batch_sampler = PaddedBCSRSampler(X, batch_size)
        train_step = create_train_step_sparse(likelihood)
    else:
        batch_sampler = CSRMatrixRowSampler(X, batch_size)
        train_step = create_train_step_dense(likelihood)

    # Convergence tracking
    best_logprob = -float("inf")
    no_improvement_count = 0

    with tqdm(range(max_epochs), desc="Training", unit="epoch", disable=quiet) as pbar:
        for epoch in pbar:
            for X_batch in batch_sampler:
                loss = train_step(model, optimizer, X_batch)
                metrics.update(neg_logprob=loss)

            logprob = -metrics.compute()["neg_logprob"]

            if not np.isfinite(logprob):
                raise ValueError("Log-likelihood is not finite")

            # Check for improvement
            if logprob - best_logprob > min_delta:
                best_logprob = logprob
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            # Update progress bar
            pbar.set_postfix(
                logprob=f"{logprob:.4f}",
                best=f"{best_logprob:.4f}",
                patience=f"{no_improvement_count}/{patience}",
            )

            metrics.reset()

            # Early stopping check
            if no_improvement_count >= patience:
                pbar.write(
                    f"Early stopping at epoch {epoch + 1}: no improvement for {patience} epochs"
                )
                break

    v = model.v_scaled()

    # Map entire X matrix through encoder in chunks
    # Use smaller chunk size for output to avoid OOM with large batch_size
    output_chunk_size = min(batch_size, 1024)
    Xnmf = np.zeros((m, k), dtype=np.float32)
    scales = np.zeros(m, dtype=np.float32)
    ll = 0.0

    # Select the appropriate likelihood function for final computation
    if likelihood == "nb":
        final_loss_fn = neg_nb_logprob_dense
    elif likelihood == "poisson":
        final_loss_fn = neg_poisson_logprob_dense
    else:
        raise ValueError(
            f"Unknown likelihood: {likelihood}. Must be 'nb' or 'poisson'."
        )

    for start_idx in range(0, m, output_chunk_size):
        end_idx = min(start_idx + output_chunk_size, m)

        X_chunk = as_dense_f32(X[start_idx:end_idx, :])
        X_chunk = jnp.array(X_chunk, dtype=jnp.float32)

        # Get encoded representation
        encoded_chunk = model.encoder(X_chunk)

        # Predict log_scale using scale encoder (same as in model.__call__)
        log_scale_chunk = model.scale_encoder(X_chunk)
        log_scale_chunk = jnp.clip(log_scale_chunk, -10.0, 10.0)

        ll += -final_loss_fn(model, X_chunk, constant_terms=True)

        Xnmf[start_idx:end_idx, :] = np.array(encoded_chunk)
        scales[start_idx:end_idx] = np.squeeze(
            np.array(jnp.exp(log_scale_chunk)), axis=1
        )

    # Store in AnnData object
    adata.obsm["X_nmf"] = Xnmf
    adata.obs["scale_nmf"] = scales
    adata.varm["V_nmf"] = np.asarray(model.v_norm()).transpose()
    adata.uns["nmf_log_likelihood"] = float(ll)

    # Filter irrelevant metagenes
    if filter_min_prop > 0:
        prop_expressed = np.mean(Xnmf > filter_min_delta, axis=0)
        keep = prop_expressed >= filter_min_prop

        if not quiet and np.sum(~keep) > 0:
            print(
                f"Filtered {np.sum(~keep)}/{k} metagenes (prop < {filter_min_prop}, delta > {filter_min_delta})"
            )

        adata.obsm["X_nmf_filtered"] = Xnmf[:, keep]
        adata.varm["V_nmf_filtered"] = adata.varm["V_nmf"][:, keep]
    else:
        adata.obsm["X_nmf_filtered"] = adata.obsm["X_nmf"]
        adata.varm["V_nmf_filtered"] = adata.varm["V_nmf"]
