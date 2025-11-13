
# Countdown

This is a fast NMF implementation for count data stored in AnnData objects,
optionally running on a GPU.

## Installing

```
pip install git+https://github.com/dcjones/countdown.git
```

or, with `uv`
```
uv add git+https://github.com/dcjones/countdown.git
```

If you intend to run this on a GPU, you may have to install jax manually first to make sure you get the cuda version, with the appropriate version of cuda.
```
pip install -U "jax[cuda13]"
```
See [the jax installation guide](https://docs.jax.dev/en/latest/installation.html).


## Usage

```python
from countdown import nmf

nmf(adata, k=100)

adata.obsm["X_nmf"] # contains a matrix of size [ncells, 100]
```

## Model

Compared to more common NMF implementations, there are a few difference geared towards usefulness in analyzing large sparse matrices of integer counts, like in transcriptomics.

  * We optimize Poisson likelihood in a simple latent factorization model:$X_{cg} \sim \text{Poisson}(U_{c\cdot} V_{\cdot g})$, where $X_{cg}$ is the observed count for cell $c$ and gene $g$, $U$ is the inferred lower-dimensional "metagene" expression matrix, and $V$ defines the metagenes as a linear combinations of genes.
  * We constrain the rows of $V$ to sum to 1. As a consequence the rows of $U$ sum approximately to the same values as the rows of $X$, essentially preserving a cell's overall expression.
  * We avoid converting a sparse count matrix into a dense matrix all at once by training on randomly sampled minibatches. To do this, a MLP probabilistic encoder is trained, in similar fashion to a VAE, to map cell count vectors to their decomposition. This is particularly useful when using a GPU with limited memory.
