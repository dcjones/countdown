
from anndata import AnnData
from flax import nnx
from scipy.sparse import csr_matrix
import jax
import jax.numpy as jnp
import numpy as np
import optax
from tqdm import tqdm

class CSRMatrixRowSampler:
    def __init__(self, X: csr_matrix, batch_size: int):
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
            self.X[self.idx[fr:to], :].todense(out=self.chunk)
            yield jnp.array(self.chunk)

class Encoder(nnx.Module):
    def __init__(self, n: int, k: int, batch_size: int, hidden_dim: int, *, rngs: nnx.Rngs):
        self.lyr1 = nnx.Linear(n, hidden_dim, rngs=rngs)
        self.lyr2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.lyr3 = nnx.Linear(hidden_dim, k, rngs=rngs)

    def __call__(self, X: jax.Array):
        u = self.lyr1(X)
        u = nnx.relu(u)
        u = self.lyr2(u)
        u = nnx.relu(u)
        u = self.lyr3(u)
        return jnp.exp(u)

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

def nmf(adata: AnnData, k: int = 64, batch_size: int = 2048, hidden_dim=128, lr=1e-3, max_epochs: int = 200, patience: int = 10, min_delta: float = 1e-4):
    m, n = adata.shape

    # X = np.asarray(adata.X.todense(), dtype=np.float32)
    # X = jnp.array(X)
    X = adata.X.astype(np.float32)

    rngs = nnx.Rngs(0)
    model = NMF(n, k, batch_size, hidden_dim, rngs=rngs)

    optimizer = nnx.Optimizer(model, optax.adam(lr))
    metrics = nnx.MultiMetric(
        neg_logprob=nnx.metrics.Average("neg_logprob")
    )

    batch_sampler = CSRMatrixRowSampler(X, batch_size)

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

    # TODO: map adata.X through encoder in chunks and store in data.obsm["Xnmf"]
