import jax
import jax.numpy as jnp
import numpy as np
import optax
from anndata import AnnData
from flax import nnx
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


class SparseEncoder(nnx.Module):
    """Encoder with sparse input support for the first layer."""

    def __init__(
        self, n: int, k: int, batch_size: int, hidden_dim: int, *, rngs: nnx.Rngs
    ):
        # First layer uses manual weights for sparse @ dense matmul
        self.weights1 = nnx.Param(
            nnx.initializers.lecun_normal()(
                rngs.params(), (n, hidden_dim // 2), jnp.float32
            )
        )
        self.bias1 = nnx.Param(jnp.zeros(hidden_dim // 2))
        self.lyr2 = nnx.Linear(hidden_dim // 2, hidden_dim, rngs=rngs)
        self.lyr3 = nnx.Linear(hidden_dim, k, rngs=rngs)
        self.ln2 = nnx.LayerNorm(hidden_dim, rngs=rngs)

    def __call__(self, X: BCSR):
        u = X @ self.weights1.value + self.bias1.value
        u = nnx.leaky_relu(u)
        u = self.lyr2(u)
        u = self.ln2(u)
        u = nnx.leaky_relu(u)
        u = self.lyr3(u)
        return nnx.softplus(u)


class DenseEncoder(nnx.Module):
    """Encoder for dense input."""

    def __init__(
        self, n: int, k: int, batch_size: int, hidden_dim: int, *, rngs: nnx.Rngs
    ):
        self.lyr1 = nnx.Linear(n, hidden_dim // 2, rngs=rngs)
        self.lyr2 = nnx.Linear(hidden_dim // 2, hidden_dim, rngs=rngs)
        self.lyr3 = nnx.Linear(hidden_dim, k, rngs=rngs)
        self.ln2 = nnx.LayerNorm(hidden_dim, rngs=rngs)

    def __call__(self, X: jax.Array):
        u = self.lyr1(X)
        u = nnx.leaky_relu(u)
        u = self.lyr2(u)
        u = self.ln2(u)
        u = nnx.leaky_relu(u)
        u = self.lyr3(u)
        return nnx.softplus(u)


class NMF(nnx.Module):
    def __init__(
        self,
        n: int,
        k: int,
        batch_size: int,
        hidden_dim: int,
        *,
        rngs: nnx.Rngs,
        sparse: bool = False,
    ):
        key = rngs.params()
        if sparse:
            self.encoder = SparseEncoder(n, k, batch_size, hidden_dim, rngs=rngs)
        else:
            self.encoder = DenseEncoder(n, k, batch_size, hidden_dim, rngs=rngs)
        self.scale = nnx.Param(jnp.zeros((1, n)))
        self.v = nnx.Param(jax.random.normal(key, (k, n)) / jnp.sqrt(n))
        self._sparse = sparse

    # X: [batch_size, n]
    def __call__(self, X: jax.Array | BCSR):
        return self.encoder(X) @ self.v_scaled()

    def v_norm(self):
        return nnx.softmax(self.v.value, axis=0)

    def v_scaled(self):
        return jnp.exp(self.scale.value) * self.v_norm()


def neg_logprob_dense(model: NMF, X: jax.Array):
    """Negative log probability for dense input."""
    λ = model(X)
    lp = (X * jnp.log(jnp.clip(λ, 1e-8))).sum() - jnp.sum(λ)
    return -lp


def neg_logprob_sparse(model: NMF, X: BCSR):
    """Negative log probability for sparse BCSR input."""
    λ = model(X)
    # Extract λ values only at non-zero positions of X
    lp = (
        X.data * jnp.log(jnp.clip(bcsr_extract(X.indices, X.indptr, λ), 1e-8))
    ).sum() - jnp.sum(λ)
    return -lp


@nnx.jit
def train_step_dense(model: NMF, optimizer: nnx.Optimizer, X: jax.Array):
    """Fused forward, backward, and optimizer update for dense input."""
    loss, grads = nnx.value_and_grad(neg_logprob_dense)(model, X)
    optimizer.update(grads)
    return loss


@nnx.jit
def train_step_sparse(model: NMF, optimizer: nnx.Optimizer, X: BCSR):
    """Fused forward, backward, and optimizer update for sparse BCSR input."""
    loss, grads = nnx.value_and_grad(neg_logprob_sparse)(model, X)
    optimizer.update(grads)
    return loss


def nmf(
    adata: AnnData,
    k: int = 64,
    batch_size: int | None = 4096,
    hidden_dim=512,
    lr=1e-3,
    max_epochs: int = 2000,
    patience: int = 40,
    min_delta: float = 1e-5,
    sparse: bool = True,
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
    lr : float, default=1e-3
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
    model = NMF(n, k, batch_size, hidden_dim, rngs=rngs, sparse=sparse)

    optimizer = nnx.Optimizer(model, optax.adam(lr))
    metrics = nnx.MultiMetric(neg_logprob=nnx.metrics.Average("neg_logprob"))

    X = adata.X

    if sparse:
        batch_sampler = PaddedBCSRSampler(X, batch_size)
        train_step = train_step_sparse
    else:
        batch_sampler = CSRMatrixRowSampler(X, batch_size)
        train_step = train_step_dense

    # Convergence tracking
    best_logprob = -float("inf")
    no_improvement_count = 0

    with tqdm(range(max_epochs), desc="Training", unit="epoch") as pbar:
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
    ll = 0.0

    # For sparse models, create a function to encode dense chunks
    # The sparse encoder's first layer is just a matmul, so we can apply it to dense input
    if sparse:

        def encode_dense(X_dense):
            u = X_dense @ model.encoder.weights1.value + model.encoder.bias1.value
            u = nnx.leaky_relu(u)
            u = model.encoder.lyr2(u)
            u = model.encoder.ln2(u)
            u = nnx.leaky_relu(u)
            u = model.encoder.lyr3(u)
            return nnx.softplus(u)

    else:
        encode_dense = model.encoder

    for start_idx in range(0, m, output_chunk_size):
        end_idx = min(start_idx + output_chunk_size, m)

        X_chunk = as_dense_f32(X[start_idx:end_idx, :])
        X_chunk = jnp.array(X_chunk, dtype=jnp.float32)
        encoded_chunk = encode_dense(X_chunk)

        λ = encoded_chunk @ v
        ll += jnp.sum(
            X_chunk * jnp.log(λ + 1e-8) - λ - jax.scipy.special.gammaln(X_chunk + 1)
        )

        Xnmf[start_idx:end_idx, :] = np.array(encoded_chunk)

    # Store in AnnData object
    adata.obsm["X_nmf"] = Xnmf
    adata.varm["V_nmf"] = np.asarray(model.v_norm()).transpose()
    adata.varm["scale_nmf"] = np.asarray(model.scale.value).squeeze()
    adata.uns["nmf_log_likelihood"] = float(ll)
