import jax
import jax.numpy as jnp
import numpy as np
import optax
from anndata import AnnData
from flax import nnx
from scipy.sparse import csr_matrix
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
        self.n = n

    def __iter__(self):
        np.random.shuffle(self.idx)
        for fr in range(0, len(self.idx), self.batch_size):
            to = min(fr + self.batch_size, self.m)
            batch_size = to - fr
            if batch_size == self.batch_size:
                self.X[self.idx[fr:to], :].todense(out=self.chunk)
                yield jnp.array(self.chunk)
            else:
                # Final partial batch - create appropriately sized array
                partial_chunk = np.zeros((batch_size, self.n), dtype=np.float32)
                self.X[self.idx[fr:to], :].todense(out=partial_chunk)
                yield jnp.array(partial_chunk)


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


class Encoder(nnx.Module):
    def __init__(
        self, n: int, k: int, batch_size: int, hidden_dim: int, *, rngs: nnx.Rngs
    ):
        self.lyr1 = nnx.Linear(n, hidden_dim, rngs=rngs)
        self.lyr2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.lyr3 = nnx.Linear(hidden_dim, k, rngs=rngs)
        self.ln2 = nnx.LayerNorm(hidden_dim, rngs=rngs)

    def __call__(self, X: jax.Array):
        u = self.lyr1(X)
        # u = nnx.tanh(u)
        u = nnx.leaky_relu(u)
        u = self.lyr2(u)
        u = self.ln2(u)
        # u = nnx.tanh(u)
        u = nnx.leaky_relu(u)
        u = self.lyr3(u)
        # jax.debug.print("u: [{}] {}", u.shape, jnp.max(u, axis=0))
        # jax.debug.print(f"u: {}", u.shape, jnp.max(u, axis=0) > )
        # u_sp = nnx.softplus(u) + 1e-3
        # u_sp = nnx.softplus(u) + 1e-3
        u_sp = jnp.exp(u)
        # jax.debug.print("{}", (jnp.max(u, axis=0) > 1e-2).sum())
        return u_sp
        # return nnx.softplus(u)


class NMF(nnx.Module):
    def __init__(
        self, n: int, k: int, batch_size: int, hidden_dim: int, *, rngs: nnx.Rngs
    ):
        key = rngs.params()
        self.encoder = Encoder(n, k, batch_size, hidden_dim, rngs=rngs)
        self.v = nnx.Param(jax.random.normal(key, (k, n)))

    # X: [batch_size, n]
    def __call__(self, X: jax.Array):
        # jax.debug.print("v: {}", jnp.max(self.v.value, axis=1))
        # v_norm = nnx.softmax(self.v.value, axis=1)

        v_pos = nnx.softplus(self.v.value) + 1e-8
        v_norm = v_pos / v_pos.sum(axis=1, keepdims=True)

        return self.encoder(jnp.log1p(X)) @ v_norm


def neg_logprob(model: NMF, X: jax.Array):
    λ = model(X)
    lp = X * jnp.log(λ + 1e-8) - λ
    # excluding the normalizing term which is expensive and constant wrt to model params
    # lp -= jax.scipy.special.gammaln(X + 1)
    return -jnp.mean(lp)


@nnx.jit
def compute_loss_and_grads(model: NMF, X: jax.Array):
    """Compute loss and gradients without applying updates."""
    grad_fn = nnx.value_and_grad(neg_logprob)
    loss, grads = grad_fn(model, X)
    return loss, grads


def zero_v_grads(grads: nnx.State) -> nnx.State:
    """Zero out V gradients, keeping only encoder gradients."""

    def zero_if_v(path, var_state):
        if "v" in path:
            return var_state.replace(value=jnp.zeros_like(var_state.value))
        return var_state

    return grads.map(zero_if_v)


def zero_encoder_grads(grads: nnx.State) -> nnx.State:
    """Zero out encoder gradients, keeping only V gradients."""

    def zero_if_encoder(path, var_state):
        if "encoder" in path:
            return var_state.replace(value=jnp.zeros_like(var_state.value))
        return var_state

    return grads.map(zero_if_encoder)


def accumulate_grads(
    acc: nnx.State | None, grads: nnx.State, weight: float
) -> nnx.State:
    """Accumulate weighted gradients."""
    weighted = grads.map(lambda path, vs: vs.replace(value=vs.value * weight))
    if acc is None:
        return weighted
    # Build flat dicts for efficient lookup
    weighted_flat = dict(weighted.flat_state())

    def add_values(path, acc_vs):
        return acc_vs.replace(value=acc_vs.value + weighted_flat[path].value)

    return acc.map(add_values)


def nmf(
    adata: AnnData,
    k: int = 64,
    batch_size: int | None = 4096,
    hidden_dim=512,
    lr=1e-3,
    max_epochs: int = 2000,
    patience: int = 40,
    min_delta: float = 1e-5,
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
    metrics = nnx.MultiMetric(neg_logprob=nnx.metrics.Average("neg_logprob"))

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
    best_logprob = -float("inf")
    no_improvement_count = 0

    # For full-batch training, use simple per-batch updates (no accumulation needed)
    use_gradient_accumulation = batch_size != m

    with tqdm(range(max_epochs), desc="Training", unit="epoch") as pbar:
        for epoch in pbar:
            if use_gradient_accumulation:
                # Accumulate V gradients across batches, update encoder per-batch
                v_grad_acc = None
                total_samples = 0

                for X_batch in batch_sampler:
                    batch_len = X_batch.shape[0]
                    loss, grads = compute_loss_and_grads(model, X_batch)

                    # Apply encoder gradients immediately (zero out V)
                    encoder_grads = zero_v_grads(grads)
                    optimizer.update(encoder_grads)

                    # Accumulate V gradients weighted by batch size
                    v_grads = zero_encoder_grads(grads)
                    v_grad_acc = accumulate_grads(v_grad_acc, v_grads, float(batch_len))
                    total_samples += batch_len

                    metrics.update(neg_logprob=loss)

                # Apply accumulated V gradient (averaged over all samples)
                if v_grad_acc is not None and total_samples > 0:
                    avg_v_grads = v_grad_acc.map(
                        lambda path, vs: vs.replace(value=vs.value / total_samples)
                    )
                    optimizer.update(avg_v_grads)
            else:
                # Full-batch training: simple per-batch update
                for X_batch in batch_sampler:
                    loss, grads = compute_loss_and_grads(model, X_batch)
                    optimizer.update(grads)
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
