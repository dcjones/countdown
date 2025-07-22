
from anndata import AnnData
from flax import nnx
from scipy.sparse import csr_matrix
import jax
import jax.numpy as jnp
import numpy as np
import optax
from tqdm import tqdm

def as_dense_f32(X: csr_matrix | np.ndarray) -> np.ndarray:
    if isinstance(X, csr_matrix):
        return X.todense().astype(np.float32)
    else:
        return X.astype(np.float32)


class CSRMatrixRowSampler:
    def __init__(self, X: csr_matrix, batch_size: int):
        m, n = X.shape
        self.X = X.astype(np.float32)
        self.idx = np.arange(m)
        self.chunk = np.zeros((batch_size, n), dtype=np.float32)
        self.batch_size = batch_size
        self.m = m

    def __iter__(self):
        np.random.shuffle(self.idx)
        for fr in range(0, len(self.idx), self.batch_size):
            to = fr + self.batch_size
            if to >= self.m:
                break
            self.X[self.idx[fr:to], :].todense(out=self.chunk)
            yield jnp.array(self.chunk)

class DenseMatrixRowSampler:
    def __init__(self, X: np.ndarray, batch_size: int):
        m, n = X.shape
        self.X = X
        self.idx = np.arange(m)
        self.chunk = np.zeros((batch_size, n), dtype=np.float32)
        self.batch_size = batch_size
        self.m = m

    def __iter__(self):
        np.random.shuffle(self.idx)
        for fr in range(0, len(self.idx), self.batch_size):
            to = fr + self.batch_size
            if to >= self.m:
                break
            self.chunk[:] = self.X[self.idx[fr:to], :]
            yield jnp.array(self.chunk)

class Encoder(nnx.Module):
    def __init__(self, n: int, k: int, batch_size: int, hidden_dim: int, *, rngs: nnx.Rngs):
        self.lyr1 = nnx.Linear(n, hidden_dim, rngs=rngs)
        self.lyr2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.lyr3 = nnx.Linear(hidden_dim, k, rngs=rngs)

    def __call__(self, X: jax.Array):
        u = self.lyr1(X)
        u = nnx.tanh(u)
        u = self.lyr2(u)
        u = nnx.tanh(u)
        u = self.lyr3(u)
        return nnx.softplus(u)

class NMF(nnx.Module):
    def __init__(self, n:int, k: int, batch_size: int, hidden_dim: int, *, rngs: nnx.Rngs):
        key = rngs.params()
        self.encoder = Encoder(n, k, batch_size, hidden_dim, rngs=rngs)
        self.v = nnx.Param(jax.random.normal(key, (k, n)))

    # X: [batch_size, n]
    def __call__(self, X: jax.Array):
        return self.encoder(X) @ nnx.softmax(self.v.value, axis=1)

def neg_logprob(model: NMF, X: jax.Array):
    λ = model(X)
    lp = X * jnp.log(λ) - λ
    # excluding the normalizing term which is expensive and constant wrt to model params
    # - gammaln(X + 1)
    return -jnp.sum(lp)

@nnx.jit
def train_step(model: NMF, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, X: jax.Array):
    grad_fn = nnx.value_and_grad(neg_logprob)
    loss, grads = grad_fn(model, X)
    metrics.update(neg_logprob=loss)
    optimizer.update(grads)

def nmf(adata: AnnData, k: int = 64, batch_size: int | None = 4096, hidden_dim=512, lr=5e-3, max_epochs: int = 2000, patience: int = 40, min_delta: float = 1e-4):
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
    hidden_dim : int, default=256
        Number of hidden units in the encoder neural network.
    lr : float, default=5e-3
        Learning rate for the Adam optimizer.
    max_epochs : int, default=2000
        Maximum number of training epochs.
    patience : int, default=40
        Number of epochs to wait for improvement before early stopping.
    min_delta : float, default=1e-4
        Minimum change in log-probability to be considered an improvement
        for early stopping.

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
    model = NMF(n, k, batch_size, hidden_dim, rngs=rngs)

    optimizer = nnx.Optimizer(model, optax.adam(lr))
    metrics = nnx.MultiMetric(
        neg_logprob=nnx.metrics.Average("neg_logprob")
    )

    X = adata.X

    if batch_size == m:
        batch_sampler = [jnp.array(as_dense_f32(X))]
    else:
        if isinstance(X, csr_matrix):
            batch_sampler = CSRMatrixRowSampler(X, batch_size)
        elif isinstance(X, np.ndarray):
            batch_sampler = DenseMatrixRowSampler(X, batch_size)
        else:
            raise ValueError(f"Unsupported data type: {type(X)}")

    # Convergence tracking
    best_logprob = -float('inf')
    no_improvement_count = 0

    with tqdm(range(max_epochs), desc="Training", unit="epoch") as pbar:
        for epoch in pbar:
            for X_batch in batch_sampler:
                train_step(model, optimizer, metrics, X_batch)

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
                patience=f"{no_improvement_count}/{patience}"
            )

            metrics.reset()

            # Early stopping check
            if no_improvement_count >= patience:
                pbar.write(f"Early stopping at epoch {epoch + 1}: no improvement for {patience} epochs")
                break

    # Map entire X matrix through encoder in chunks
    Xnmf = np.zeros((m, k), dtype=np.float32)
    for start_idx in range(0, m, batch_size):
        end_idx = min(start_idx + batch_size, m)

        X_chunk = as_dense_f32(X[start_idx:end_idx, :])
        X_chunk = jnp.array(X_chunk, dtype=jnp.float32)
        encoded_chunk = model.encoder(X_chunk)

        Xnmf[start_idx:end_idx, :] = np.array(encoded_chunk)

    # Store in AnnData object
    adata.obsm["X_nmf"] = Xnmf
