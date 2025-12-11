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
    Simple single-layer encoder that outputs both NMF factors U and log-scale.
    """

    def __init__(self, n: int, k: int, *, rngs: nnx.Rngs):
        self.weights = nnx.Param(
            nnx.initializers.lecun_normal()(rngs.params(), (n, k), jnp.float32)
        )
        self.bias = nnx.Param(jnp.zeros(k))

        # Separate layer for log-scale prediction
        # Initialize with small weights so log_scale starts near 0 (scale near 1)
        self.scale_weights = nnx.Param(
            1e-6 * jax.random.normal(rngs.params(), (n, 1), dtype=jnp.float32)
        )
        self.scale_bias = nnx.Param(jnp.zeros(1))

    def __call__(self, X: jax.Array | BCSR):
        u = nnx.softplus(X @ self.weights.value + self.bias.value)

        log_scale = X @ self.scale_weights.value + self.scale_bias.value
        return u, log_scale


class Encoder(nnx.Module):
    """Encoder that supports both sparse (BCSR) and dense inputs, outputs U and log-scale."""

    def __init__(self, n: int, k: int, hidden_dim: int, *, rngs: nnx.Rngs):
        # First layer uses manual weights to support sparse @ dense matmul
        self.weights1 = nnx.Param(
            nnx.initializers.lecun_normal()(
                rngs.params(), (n, hidden_dim // 2), jnp.float32
            )
        )
        self.bias1 = nnx.Param(jnp.zeros(hidden_dim // 2))

        self.weights_shortcut = nnx.Param(
            nnx.initializers.lecun_normal()(rngs.params(), (n, k), jnp.float32)
        )
        self.bias_shortcut = nnx.Param(jnp.zeros(k))

        self.lyr2 = nnx.Linear(hidden_dim // 2, hidden_dim, rngs=rngs)
        self.lyr3 = nnx.Linear(hidden_dim, k, rngs=rngs)

        self.ln1 = nnx.LayerNorm(hidden_dim // 2, rngs=rngs)

        # Separate layer for scale prediction
        self.scale_lyr1 = nnx.Linear(hidden_dim, 1, rngs=rngs)

    def __call__(self, X: jax.Array | BCSR):
        # residual = X @ self.weights_shortcut.value + self.bias_shortcut.value
        u = X @ self.weights1.value + self.bias1.value
        u = self.ln1(u)
        u = nnx.leaky_relu(u)
        u = self.lyr2(u)
        u = nnx.leaky_relu(u)
        log_scale = self.scale_lyr1(u)

        u = self.lyr3(u)
        u = nnx.softplus(u)  # + nnx.softplus(residual)

        return u, log_scale


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
        simple_encoder: bool = True,
    ):
        key = rngs.params()
        if simple_encoder:
            self.encoder = SimpleEncoder(n, k, rngs=rngs)
        else:
            self.encoder = Encoder(n, k, hidden_dim=hidden_dim, rngs=rngs)
        self.v = nnx.Param(jax.random.normal(key, (k, n)) / jnp.sqrt(n))
        self.log_r = nnx.Param(jnp.full(n, 1e-1))

        # Store prior hyperparameters (not trainable)
        self.r_prior_alpha = r_prior_alpha
        self.r_prior_beta = r_prior_beta
        self.scale_prior_sigma = scale_prior_sigma

    # X: [batch_size, n]
    def __call__(self, X: jax.Array | BCSR) -> jax.Array:
        u, log_scale = self.encoder(X)
        scale = jnp.exp(log_scale)  # [batch_size]

        # jax.debug.print("scale: [{}, {}]", jnp.min(scale), jnp.max(scale))

        lambda_base = u @ self.v_scaled()  # [batch_size, n]
        # Scale each cell's predictions by its scale factor
        lambda_scaled = lambda_base * scale  # [batch_size, n]
        # lambda_scaled = lambda_base

        return lambda_scaled, scale

    def v_norm(self) -> jax.Array:
        return nnx.softmax(self.v.value, axis=1)

    def v_scaled(self) -> jax.Array:
        return self.v_norm()

    def r(self) -> jax.Array:
        return jnp.exp(self.log_r.value)

    def log_prior(self, log_scale: jax.Array) -> jax.Array:
        """
        Compute log prior for all parameters.

        Currently implements:
        - Gamma(alpha, beta) prior on dispersion parameters r
        - Normal(0, sigma) prior on scale_logit (cell-specific scale factors)

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

        return neg_log_prior


def neg_poisson_logprob_dense(model: NMF, X: jax.Array, constant_terms: bool = False):
    """Negative log probability for dense input."""
    λ, log_scale = model(X)
    lp = (X * jnp.log(jnp.clip(λ, 1e-8))).sum() - jnp.sum(λ)

    # constant wrt to parameters
    if constant_terms:
        lp -= jnp.sum(jax.scipy.special.gammaln(X + 1))

    neg_log_likelihood = -lp

    # Add negative log prior (for MAP estimation)
    neg_log_posterior = neg_log_likelihood + model.log_prior(log_scale)

    return neg_log_posterior


def neg_poisson_logprob_sparse(model: NMF, X: BCSR, constant_terms: bool = False):
    """Negative log probability for sparse BCSR input."""
    λ, log_scale = model(X)
    # Extract λ values only at non-zero positions of X
    lp = (
        X.data * jnp.log(jnp.clip(bcsr_extract(X.indices, X.indptr, λ), 1e-8))
    ).sum() - jnp.sum(λ)

    # constant wrt to parameters
    if constant_terms:
        lp -= jnp.sum(jax.scipy.special.gammaln(X.data + 1))

    neg_log_likelihood = -lp

    # Add negative log prior (for MAP estimation)
    neg_log_posterior = neg_log_likelihood + model.log_prior(log_scale)

    return neg_log_posterior


def neg_nb_logprob_dense(model: NMF, X: jax.Array, constant_terms: bool = False):
    λ, log_scale = model(X)  # [ncells, ngenes], [ncells]
    r = jnp.expand_dims(model.r(), 0)  # [1, ngenes]
    log_r = jnp.expand_dims(model.log_r.value, 0)  # [1, ngenes]
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
    neg_log_posterior = neg_log_likelihood + model.log_prior(log_scale)

    return neg_log_posterior


def neg_nb_logprob_sparse(model: NMF, X: BCSR, constant_terms: bool = False):
    λ, log_scale = model(X)  # [ncells, ngenes], [ncells]
    r = jnp.expand_dims(model.r(), 0)  # [1, ngenes]
    log_r = jnp.expand_dims(model.log_r.value, 0)  # [1, ngenes]
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
    neg_log_posterior = neg_log_likelihood + model.log_prior(log_scale)

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
        optimizer.update(grads)
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
        optimizer.update(grads)
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
    simple_encoder: bool = True,
    quiet: bool = False,
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

    Returns
    -------
    None
        The function modifies the input AnnData object in-place.

    Notes
    -----
    Results are stored in the AnnData object as:

    - `adata.obsm["X_nmf"]` : ndarray of shape (n_observations, k)
        The low-dimensional NMF representation of the observations.
        Each row corresponds to an observation (cell) and each column
        to a latent factor.

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

    rngs = nnx.Rngs(0)
    model = NMF(
        n,
        k,
        hidden_dim,
        rngs=rngs,
        r_prior_alpha=r_prior_alpha,
        r_prior_beta=r_prior_beta,
        scale_prior_sigma=scale_prior_sigma,
        simple_encoder=simple_encoder,
    )

    optimizer = nnx.Optimizer(model, optax.adam(lr))
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
        encoded_chunk, log_scale_chunk = model.encoder(X_chunk)

        ll += -final_loss_fn(model, X_chunk, constant_terms=True)

        Xnmf[start_idx:end_idx, :] = np.array(encoded_chunk)
        scales[start_idx:end_idx] = np.squeeze(
            np.array(jnp.exp(log_scale_chunk)), axis=1
        )

    # Store in AnnData object
    adata.obsm["X_nmf"] = Xnmf
    adata.obs["scale_nmf"] = scales
    adata.varm["V_nmf"] = np.asarray(model.v_norm()).transpose()
    # adata.varm["scale_nmf"] = np.asarray(model.scale.value).squeeze()
    adata.uns["nmf_log_likelihood"] = float(ll)
