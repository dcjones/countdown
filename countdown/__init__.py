
from anndata import AnnData
import jax
import jax.numpy as jnp
import optax
import numpy as np
from flax import nnx

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

# @nnx.jit
def train_step(model: NMF, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, X: jax.Array):
    grad_fn = nnx.value_and_grad(neg_logprob)
    loss, grads = grad_fn(model, X)
    metrics.update(neg_logprob=loss)
    optimizer.update(grads)

def nmf(adata: AnnData, k: int = 64, batch_size: int = 1024, hidden_dim=128, lr=1e-3):
    m, n = adata.shape

    X = np.asarray(adata.X.todense(), dtype=np.float32)
    X = jnp.array(X)

    rngs = nnx.Rngs(0)
    model = NMF(n, k, batch_size, hidden_dim, rngs=rngs)
    print(model)

    optimizer = nnx.Optimizer(model, optax.adam(lr))
    metrics = nnx.MultiMetric(
        neg_logprob=nnx.metrics.Average("neg_logprob")
    )

    for step in range(100):
        print(step)
        train_step(model, optimizer, metrics, X)
        print(metrics.compute())
        metrics.reset()

    # TODO: map adata.X through encoder in chunks and store in data.obsm["Xnmf"]
