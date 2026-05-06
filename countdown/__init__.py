import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from anndata import AnnData
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from tqdm import tqdm


def as_dense_f32(X: csr_matrix | np.ndarray) -> np.ndarray:
    if isinstance(X, csr_matrix):
        return np.asarray(X.todense()).astype(np.float32)
    else:
        return X.astype(np.float32)


class SparseBatchSampler:
    """
    Samples batches of rows from a CSR matrix as torch sparse_csr_tensor objects.

    Precomputes all batches at initialization as pinned CPU tensors (including
    row indices for the loss function), then transfers to GPU during iteration
    using non-blocking transfers for better CPU/GPU overlap.
    """

    def __init__(self, X: csr_matrix | np.ndarray, batch_size: int, device: torch.device):
        if not isinstance(X, csr_matrix):
            from scipy.sparse import csr_matrix as make_csr
            X = make_csr(X)
        m, n = X.shape
        X = X.astype(np.float32)
        self.n = n
        self.device = device
        use_pin = device.type == "cuda"

        def _maybe_pin(t: torch.Tensor) -> torch.Tensor:
            return t.pin_memory() if use_pin else t

        idx = np.arange(m)
        np.random.shuffle(idx)

        self._batches: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]] = []
        for fr in range(0, m, batch_size):
            to = min(fr + batch_size, m)
            batch_idx = idx[fr:to].copy()
            batch_idx.sort()
            sliced = X[batch_idx, :]

            nnz_per_row = np.diff(sliced.indptr)
            row_idx_np = np.repeat(np.arange(sliced.shape[0], dtype=np.int64), nnz_per_row)

            self._batches.append((
                _maybe_pin(torch.from_numpy(sliced.data.copy())),
                _maybe_pin(torch.from_numpy(sliced.indices.astype(np.int64))),
                _maybe_pin(torch.from_numpy(sliced.indptr.astype(np.int64))),
                _maybe_pin(torch.from_numpy(row_idx_np)),
                sliced.shape[0],
            ))

    def __iter__(self):
        nb = self.device.type == "cuda"
        with torch.sparse.check_sparse_tensor_invariants(enable=False):
            for data, indices, indptr, row_idx, batch_m in self._batches:
                crow = indptr.to(self.device, non_blocking=nb)
                col  = indices.to(self.device, non_blocking=nb)
                vals = data.to(self.device, non_blocking=nb)
                row  = row_idx.to(self.device, non_blocking=nb)
                yield (
                    torch.sparse_csr_tensor(
                        crow, col, vals,
                        size=(batch_m, self.n),
                        dtype=torch.float32,
                        device=self.device,
                    ),
                    row,
                )


class DenseRowSampler:
    """Samples random batches of rows from a CSR or dense matrix, yielding dense tensors."""

    def __init__(self, X: csr_matrix | np.ndarray, batch_size: int, device: torch.device):
        if isinstance(X, csr_matrix):
            X = np.asarray(X.todense()).astype(np.float32)
        else:
            X = X.astype(np.float32)
        self.X = X
        m, _ = X.shape
        self.idx = np.arange(m)
        self.batch_size = batch_size
        self.m = m
        self.device = device

    def __iter__(self):
        np.random.shuffle(self.idx)
        for fr in range(0, self.m, self.batch_size):
            to = min(fr + self.batch_size, self.m)
            batch = self.X[self.idx[fr:to], :]
            yield torch.tensor(batch, dtype=torch.float32, device=self.device), None


class SparseLinear(nn.Module):
    """
    Linear layer that handles both sparse CSR and dense input.
    Weight is stored as [in_features, out_features] for torch.sparse.mm compatibility.
    Initialized with LeCun normal (std = sqrt(1/fan_in)).
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        std = math.sqrt(1.0 / in_features)
        nn.init.normal_(self.weight, 0.0, std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.layout == torch.sparse_csr:
            return torch.sparse.mm(x, self.weight) + self.bias
        return x @ self.weight + self.bias


class SimpleEncoder(nn.Module):
    """
    Simple single-layer encoder: X -> linear -> softplus -> U

    Architecture:
    - Single linear projection from input to factors
    - Separate linear layer for scale prediction
    - Fast, low correlation, good baseline

    Best for: Quick results, interpretability, low metagene correlation
    """

    def __init__(self, n: int, k: int):
        super().__init__()
        self.layer = SparseLinear(n, k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.layer(x))


class DeepSoftplusEncoder(nn.Module):
    """
    Two-layer encoder with softplus activations: X -> hidden -> output

    Architecture:
    - Input -> hidden_dim (softplus) -> k (softplus)
    - Softplus activations work well with count data
    - Scale predicted directly from input

    Best for: Maximum likelihood with regularization
    Recommended with: metagene_reg_type="correlation", metagene_reg_strength=0.01
    """

    def __init__(self, n: int, k: int, hidden_dim: int):
        super().__init__()
        self.layer1 = SparseLinear(n, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, k)
        std = math.sqrt(1.0 / hidden_dim)
        nn.init.normal_(self.layer2.weight, 0.0, std)
        nn.init.zeros_(self.layer2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.softplus(self.layer1(x))
        return F.softplus(self.layer2(h))


class DeepCompactEncoder(nn.Module):
    """
    Compact two-layer encoder: X -> k-dim hidden -> k output

    Architecture:
    - Input -> k (softplus) -> k (softplus)
    - Reduced capacity vs DeepSoftplusEncoder (k vs hidden_dim)
    - Scale predicted directly from input

    Best for: Good likelihood with lower capacity, works well with regularization
    """

    def __init__(self, n: int, k: int, hidden_dim: int):  # noqa: ARG002
        super().__init__()
        # Uses k as hidden dim, ignoring hidden_dim, to match SimpleEncoder capacity
        self.layer1 = SparseLinear(n, k)
        self.layer2 = nn.Linear(k, k)
        std = math.sqrt(1.0 / k)
        nn.init.normal_(self.layer2.weight, 0.0, std)
        nn.init.zeros_(self.layer2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.softplus(self.layer1(x))
        return F.softplus(self.layer2(h))


class BoundedAuxiliaryEncoder(nn.Module):
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

    def __init__(self, n: int, k: int, hidden_dim: int):
        super().__init__()
        self.direct = SparseLinear(n, k)
        self.aux = SparseLinear(n, k)
        # Scale aux weights down by 0.1 to match JAX initialization
        with torch.no_grad():
            self.aux.weight.mul_(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u_direct = self.direct(x)
        u_aux = torch.tanh(self.aux(x))
        return F.softplus(u_direct + u_aux)


class ScaleEncoder(nn.Module):
    """
    Simple MLP encoder for predicting cell-specific scale factors.

    Architecture:
    - Input (n genes) -> hidden layer (32 units) -> softplus -> output (1 scale)
    - Small hidden layer for stability with high-dimensional sparse input
    - Predicts log_scale which is clipped and exponentiated

    Trained jointly with main encoder to provide stable scale predictions.
    """

    def __init__(self, n: int):
        super().__init__()
        hidden_dim = 32
        self.layer1 = SparseLinear(n, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, 1)
        # Small initialization for stability (matches JAX: 1e-3 * normal)
        nn.init.normal_(self.layer1.weight, 0.0, 1e-3)
        nn.init.normal_(self.layer2.weight, 0.0, 1e-3)
        nn.init.zeros_(self.layer2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.softplus(self.layer1(x))
        return self.layer2(h)


class NMF(nn.Module):
    def __init__(
        self,
        n: int,
        k: int,
        hidden_dim: int,
        r_prior_alpha: float = 2.0,
        r_prior_beta: float = 2.0,
        scale_prior_sigma: float = 0.5,
        encoder_version: str = "simple",
        metagene_reg_type: str = "none",
        metagene_reg_strength: float = 0.01,
        gene_scale_factors: bool = False,
        init_method: str = "normal",
        v_init: np.ndarray | None = None,
    ):
        super().__init__()

        if encoder_version == "simple":
            self.encoder = SimpleEncoder(n, k)
        elif encoder_version == "deep_softplus":
            self.encoder = DeepSoftplusEncoder(n, k, hidden_dim)
        elif encoder_version == "deep_compact":
            self.encoder = DeepCompactEncoder(n, k, hidden_dim)
        elif encoder_version == "bounded_auxiliary":
            self.encoder = BoundedAuxiliaryEncoder(n, k, hidden_dim)
        else:
            raise ValueError(
                f"Unknown encoder_version: {encoder_version}. "
                f"Must be one of: 'simple', 'deep_softplus', 'deep_compact', 'bounded_auxiliary'"
            )

        if v_init is not None:
            self.v = nn.Parameter(torch.tensor(v_init, dtype=torch.float32))
        elif init_method == "normal":
            v = torch.empty(k, n)
            nn.init.normal_(v, std=1.0 / math.sqrt(n))
            self.v = nn.Parameter(v)
        elif init_method == "uniform":
            v = torch.empty(k, n)
            nn.init.uniform_(v, -1.0 / math.sqrt(n), 1.0 / math.sqrt(n))
            self.v = nn.Parameter(v)
        elif init_method == "kaiming":
            v = torch.empty(k, n)
            nn.init.kaiming_normal_(v)
            self.v = nn.Parameter(v)
        elif init_method == "xavier":
            v = torch.empty(k, n)
            nn.init.xavier_normal_(v)
            self.v = nn.Parameter(v)
        elif init_method == "orthogonal":
            v = torch.empty(k, n)
            nn.init.orthogonal_(v)
            self.v = nn.Parameter(v)
        else:
            raise ValueError(f"Unknown init_method: {init_method}")

        self.log_r = nn.Parameter(torch.full((n,), 1e-1))
        self.scale_encoder = ScaleEncoder(n)

        # Store prior hyperparameters (not trainable)
        self.r_prior_alpha = r_prior_alpha
        self.r_prior_beta = r_prior_beta
        self.scale_prior_sigma = scale_prior_sigma
        self.metagene_reg_type = metagene_reg_type
        self.metagene_reg_strength = metagene_reg_strength
        self.gene_scale_factors = gene_scale_factors

    # X: [batch_size, n]
    def forward(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if X.layout == torch.sparse_csr:
            u, log_scale = self._fused_sparse_forward(X)
        else:
            u = self.encoder(X)
            log_scale = self.scale_encoder(X)
        log_scale = log_scale.clamp(-10.0, 10.0)
        scale = torch.exp(log_scale)                 # [batch_size, 1]
        lambda_base = u @ self.v_scaled()            # [batch_size, n]
        if self.gene_scale_factors:
            lambda_scaled = lambda_base * scale
        else:
            lambda_scaled = lambda_base
        return lambda_scaled, u, log_scale

    def _fused_sparse_forward(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Fuse all first-layer sparse matrix multiplications into one torch.sparse.mm call.
        This reads X from memory only once instead of once per SparseLinear layer,
        reducing memory bandwidth pressure for large sparse inputs.
        """
        enc = self.encoder
        s1 = self.scale_encoder.layer1

        if isinstance(enc, SimpleEncoder):
            fused_w = torch.cat([enc.layer.weight, s1.weight], dim=1)
            fused_b = torch.cat([enc.layer.bias, s1.bias])
            k = enc.layer.weight.shape[1]
            h = torch.sparse.mm(X, fused_w) + fused_b
            u = F.softplus(h[:, :k])
            log_scale = self.scale_encoder.layer2(F.softplus(h[:, k:]))

        elif isinstance(enc, BoundedAuxiliaryEncoder):
            fused_w = torch.cat([enc.direct.weight, enc.aux.weight, s1.weight], dim=1)
            fused_b = torch.cat([enc.direct.bias, enc.aux.bias, s1.bias])
            k = enc.direct.weight.shape[1]
            h = torch.sparse.mm(X, fused_w) + fused_b
            u = F.softplus(h[:, :k] + torch.tanh(h[:, k:2*k]))
            log_scale = self.scale_encoder.layer2(F.softplus(h[:, 2*k:]))

        elif isinstance(enc, (DeepSoftplusEncoder, DeepCompactEncoder)):
            fused_w = torch.cat([enc.layer1.weight, s1.weight], dim=1)
            fused_b = torch.cat([enc.layer1.bias, s1.bias])
            hd = enc.layer1.weight.shape[1]
            h = torch.sparse.mm(X, fused_w) + fused_b
            u = F.softplus(enc.layer2(F.softplus(h[:, :hd])))
            log_scale = self.scale_encoder.layer2(F.softplus(h[:, hd:]))

        else:
            u = enc(X)
            log_scale = self.scale_encoder(X)

        return u, log_scale

    def v_norm(self) -> torch.Tensor:
        return F.softmax(self.v, dim=1)

    def v_scaled(self) -> torch.Tensor:
        return self.v_norm()

    def r(self) -> torch.Tensor:
        return torch.exp(self.log_r)

    def metagene_regularization(self, u: torch.Tensor) -> torch.Tensor:
        """
        Compute regularization term to encourage diverse/orthogonal metagenes.

        Args:
            u: [batch_size, k] metagene usage matrix from current batch

        Returns penalty term (to be minimized).
        """
        if self.metagene_reg_type == "none":
            return torch.zeros(1, device=self.v.device).squeeze()

        v_norm = self.v_norm()  # [k, n]

        if self.metagene_reg_type == "correlation":
            v_centered = v_norm - v_norm.mean(dim=1, keepdim=True)
            row_norms = torch.sqrt((v_centered ** 2).sum(dim=1, keepdim=True))
            v_normalized = v_centered / (row_norms + 1e-8)
            corr_matrix = v_normalized @ v_normalized.T  # [k, k]
            k = corr_matrix.shape[0]
            mask = 1.0 - torch.eye(k, device=v_norm.device)
            penalty = ((corr_matrix * mask) ** 2).sum()
            return self.metagene_reg_strength * penalty

        elif self.metagene_reg_type == "orthogonal":
            gram = v_norm @ v_norm.T  # [k, k]
            k = gram.shape[0]
            identity = torch.eye(k, device=v_norm.device)
            penalty = ((gram - identity) ** 2).sum()
            return self.metagene_reg_strength * penalty

        elif self.metagene_reg_type == "cosine":
            row_norms = torch.sqrt((v_norm ** 2).sum(dim=1, keepdim=True))
            v_unit = v_norm / (row_norms + 1e-8)
            cosine_sim = v_unit @ v_unit.T  # [k, k]
            k = cosine_sim.shape[0]
            mask = 1.0 - torch.eye(k, device=v_norm.device)
            penalty = ((cosine_sim * mask) ** 2).sum()
            return self.metagene_reg_strength * penalty

        elif self.metagene_reg_type == "entropy":
            u_mean = u.mean(dim=0)  # [k]
            u_prob = u_mean / (u_mean.sum() + 1e-8)
            entropy = -(u_prob * torch.log(u_prob + 1e-8)).sum()
            max_entropy = math.log(float(u.shape[1]))
            penalty = max_entropy - entropy
            return self.metagene_reg_strength * penalty

        elif self.metagene_reg_type == "usage":
            threshold = 0.1
            active = torch.sigmoid(100.0 * (u - threshold)).mean(dim=0)  # [k]
            penalty = torch.clamp(0.01 - active, min=0.0).sum()
            return self.metagene_reg_strength * penalty

        elif self.metagene_reg_type == "weighted_correlation":
            gene_variance = v_norm.var(dim=0)  # [n]
            gene_weights = gene_variance / (gene_variance.sum() + 1e-8)
            v_weighted = v_norm * torch.sqrt(gene_weights)
            v_centered = v_weighted - v_weighted.mean(dim=1, keepdim=True)
            row_norms = torch.sqrt((v_centered ** 2).sum(dim=1, keepdim=True))
            v_normalized = v_centered / (row_norms + 1e-8)
            corr_matrix = v_normalized @ v_normalized.T
            k = corr_matrix.shape[0]
            mask = 1.0 - torch.eye(k, device=v_norm.device)
            penalty = ((corr_matrix * mask) ** 2).sum()
            return self.metagene_reg_strength * penalty

        elif self.metagene_reg_type == "combined":
            # Weighted correlation
            gene_variance = v_norm.var(dim=0)
            gene_weights = gene_variance / (gene_variance.sum() + 1e-8)
            v_weighted = v_norm * torch.sqrt(gene_weights)
            v_centered = v_weighted - v_weighted.mean(dim=1, keepdim=True)
            row_norms = torch.sqrt((v_centered ** 2).sum(dim=1, keepdim=True))
            v_normalized = v_centered / (row_norms + 1e-8)
            corr_matrix = v_normalized @ v_normalized.T
            k = corr_matrix.shape[0]
            mask = 1.0 - torch.eye(k, device=v_norm.device)
            corr_penalty = ((corr_matrix * mask) ** 2).sum()
            # Entropy
            u_mean = u.mean(dim=0)
            u_prob = u_mean / (u_mean.sum() + 1e-8)
            entropy = -(u_prob * torch.log(u_prob + 1e-8)).sum()
            max_entropy = math.log(float(u.shape[1]))
            entropy_penalty = max_entropy - entropy
            return self.metagene_reg_strength * (corr_penalty + entropy_penalty)

        elif self.metagene_reg_type == "balanced":
            # Weighted correlation
            gene_variance = v_norm.var(dim=0)
            gene_weights = gene_variance / (gene_variance.sum() + 1e-8)
            v_weighted = v_norm * torch.sqrt(gene_weights)
            v_centered = v_weighted - v_weighted.mean(dim=1, keepdim=True)
            row_norms = torch.sqrt((v_centered ** 2).sum(dim=1, keepdim=True))
            v_normalized = v_centered / (row_norms + 1e-8)
            corr_matrix = v_normalized @ v_normalized.T
            k = corr_matrix.shape[0]
            mask = 1.0 - torch.eye(k, device=v_norm.device)
            corr_penalty = ((corr_matrix * mask) ** 2).sum()
            # Entropy with higher weight to force usage
            u_mean = u.mean(dim=0)
            u_prob = u_mean / (u_mean.sum() + 1e-8)
            entropy = -(u_prob * torch.log(u_prob + 1e-8)).sum()
            max_entropy = math.log(float(u.shape[1]))
            entropy_penalty = max_entropy - entropy
            return self.metagene_reg_strength * (corr_penalty + 1_000_000.0 * entropy_penalty)

        else:
            return torch.zeros(1, device=self.v.device).squeeze()

    def log_prior(self, u: torch.Tensor, log_scale: torch.Tensor) -> torch.Tensor:
        """
        Compute log prior for all parameters.

        Currently implements:
        - Gamma(alpha, beta) prior on dispersion parameters r
        - Normal(0, sigma) prior on scale_logit (cell-specific scale factors)
        - Optional metagene regularization (correlation, orthogonal, or cosine)

        Args:
            u: [batch_size, k] metagene usage for current batch
            log_scale: [batch_size, 1] log scale values for current batch

        Returns negative log prior (to be minimized).
        """
        r = self.r()
        alpha = self.r_prior_alpha
        beta = self.r_prior_beta

        log_prior_r = (
            alpha * math.log(beta)
            - math.lgamma(alpha)
            + (alpha - 1) * torch.log(r)
            - beta * r
        )
        neg_log_prior = -log_prior_r.sum()

        sigma = self.scale_prior_sigma
        log_prior_scale = (
            -0.5 * math.log(2 * math.pi)
            - math.log(sigma)
            - log_scale ** 2 / (2 * sigma ** 2)
        )
        neg_log_prior = neg_log_prior - log_prior_scale.sum()

        neg_log_prior = neg_log_prior + self.metagene_regularization(u)

        return neg_log_prior


def _sparse_row_col_indices(
    X: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (row_idx, col_idx) arrays indexing every non-zero element in a sparse_csr_tensor."""
    crow = X.crow_indices()
    col_idx = X.col_indices()
    batch_size = X.shape[0]
    row_idx = torch.repeat_interleave(
        torch.arange(batch_size, device=X.device),
        crow[1:] - crow[:-1],
    )
    return row_idx, col_idx


def neg_poisson_logprob_dense(model: NMF, X: torch.Tensor, constant_terms: bool = False, row_idx: torch.Tensor | None = None):
    """Negative log posterior for dense input under Poisson likelihood."""
    λ, u, log_scale = model(X)
    lp = (X * torch.log(λ.clamp(1e-8))).sum() - λ.sum()
    if constant_terms:
        lp -= torch.lgamma(X + 1).sum()
    return -lp + model.log_prior(u, log_scale)


def neg_poisson_logprob_sparse(model: NMF, X: torch.Tensor, constant_terms: bool = False, row_idx: torch.Tensor | None = None):
    """Negative log posterior for sparse CSR input under Poisson likelihood."""
    λ, u, log_scale = model(X)
    col_idx = X.col_indices()
    if row_idx is None:
        row_idx, col_idx = _sparse_row_col_indices(X)
    x_data = X.values()
    lp = (x_data * torch.log(λ[row_idx, col_idx].clamp(1e-8))).sum() - λ.sum()
    if constant_terms:
        lp -= torch.lgamma(x_data + 1).sum()
    return -lp + model.log_prior(u, log_scale)


def neg_nb_logprob_dense(model: NMF, X: torch.Tensor, constant_terms: bool = False, row_idx: torch.Tensor | None = None):
    """Negative log posterior for dense input under Negative Binomial likelihood."""
    λ, u, log_scale = model(X)
    r = model.r().unsqueeze(0)        # [1, n]
    log_r = model.log_r.unsqueeze(0)  # [1, n]
    log_λr = torch.log(λ + r)
    log_λ = torch.log(λ)

    ncells = X.shape[0]

    lp = -(ncells * torch.lgamma(r)).sum()
    lp += (ncells * r * log_r).sum()
    lp -= (r * log_λr.sum(dim=0, keepdim=True)).sum()
    lp += (X * (log_λ - log_λr)).sum()
    lp += torch.lgamma(X + r).sum()

    if constant_terms:
        lp -= torch.lgamma(X + 1).sum()

    return -lp + model.log_prior(u, log_scale)


def neg_nb_logprob_sparse(model: NMF, X: torch.Tensor, constant_terms: bool = False, row_idx: torch.Tensor | None = None):
    """Negative log posterior for sparse CSR input under Negative Binomial likelihood."""
    λ, u, log_scale = model(X)
    r = model.r().unsqueeze(0)        # [1, n]
    log_r = model.log_r.unsqueeze(0)  # [1, n]
    log_λr = torch.log(λ + r)
    log_λ = torch.log(λ)

    ncells = X.shape[0]
    col_idx = X.col_indices()
    if row_idx is None:
        row_idx, col_idx = _sparse_row_col_indices(X)
    x_data = X.values()

    lp = (ncells * r * log_r).sum()
    lp -= (r * log_λr.sum(dim=0, keepdim=True)).sum()

    # X * (log_λ - log_λr) at non-zero positions only
    log_diff = log_λ - log_λr
    lp += (x_data * log_diff[row_idx, col_idx]).sum()

    # gammaln(X + r) - gammaln(r) at non-zero positions only.
    # At zero positions, lgamma(0 + r) - lgamma(r) = 0, so these contribute nothing.
    r_col = r[0, col_idx]
    lp -= torch.lgamma(r_col).sum()
    lp += torch.lgamma(r_col + x_data).sum()

    if constant_terms:
        lp -= torch.lgamma(x_data + 1).sum()

    return -lp + model.log_prior(u, log_scale)


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
        If True, use sparse CSR batches (memory efficient, allows larger batches).
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
    init_ncells : int, default=10000
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
    m: int = adata.shape[0]
    n: int = adata.shape[1]

    if batch_size is None:
        batch_size = m
    batch_size = int(batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    v_init = None
    if init_method == "nndsvd":
        if not quiet:
            print("Computing NNDSVD initialization...")

        X_init = adata.X
        if m > init_ncells:
            if not quiet:
                print(f"Subsampling {init_ncells} cells for initialization...")
            rng = np.random.default_rng(42)
            indices = rng.choice(m, init_ncells, replace=False)
            indices.sort()
            X_init = X_init[indices]

        U, S, Vt = svds(X_init, k=k)
        idx = np.argsort(S)[::-1]
        S = S[idx]
        U = U[:, idx]
        Vt = Vt[idx, :]

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

        H = H + 1e-6
        H_norm = H / H.sum(axis=1, keepdims=True)
        v_init = np.log(H_norm).astype(np.float32)

    model = NMF(
        n,
        k,
        hidden_dim,
        r_prior_alpha=r_prior_alpha,
        r_prior_beta=r_prior_beta,
        scale_prior_sigma=scale_prior_sigma,
        encoder_version=encoder_version,
        metagene_reg_type=metagene_reg_type,
        metagene_reg_strength=metagene_reg_strength,
        gene_scale_factors=gene_scale_factors,
        init_method=init_method,
        v_init=v_init,
    ).to(device)

    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif optimizer_name == "rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    if likelihood == "nb":
        loss_fn_sparse = neg_nb_logprob_sparse
        loss_fn_dense = neg_nb_logprob_dense
    elif likelihood == "poisson":
        loss_fn_sparse = neg_poisson_logprob_sparse
        loss_fn_dense = neg_poisson_logprob_dense
    else:
        raise ValueError(
            f"Unknown likelihood: {likelihood}. Must be 'nb' or 'poisson'."
        )

    X = adata.X

    if sparse:
        batch_sampler = SparseBatchSampler(X, batch_size, device)
        train_loss_fn = loss_fn_sparse
    else:
        batch_sampler = DenseRowSampler(X, batch_size, device)
        train_loss_fn = loss_fn_dense

    # Convergence tracking
    best_logprob = -float("inf")
    no_improvement_count = 0

    model.train()
    with tqdm(range(max_epochs), desc="Training", unit="epoch", disable=quiet) as pbar:
        for epoch in pbar:
            epoch_loss_sum = 0.0
            epoch_batch_count = 0

            for X_batch, precomputed_row_idx in batch_sampler:
                optimizer.zero_grad(set_to_none=True)
                loss = train_loss_fn(model, X_batch, row_idx=precomputed_row_idx)
                loss.backward()
                optimizer.step()
                epoch_loss_sum += loss.detach().item()
                epoch_batch_count += 1

            logprob = -(epoch_loss_sum / epoch_batch_count)

            if not np.isfinite(logprob):
                raise ValueError("Log-likelihood is not finite")

            if logprob - best_logprob > min_delta:
                best_logprob = logprob
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            pbar.set_postfix(
                logprob=f"{logprob:.4f}",
                best=f"{best_logprob:.4f}",
                patience=f"{no_improvement_count}/{patience}",
            )

            if no_improvement_count >= patience:
                pbar.write(
                    f"Early stopping at epoch {epoch + 1}: no improvement for {patience} epochs"
                )
                break

    # Map entire X matrix through encoder in chunks
    output_chunk_size = min(batch_size, 1024)
    Xnmf = np.zeros((m, k), dtype=np.float32)
    scales = np.zeros(m, dtype=np.float32)
    ll = 0.0

    model.eval()
    with torch.no_grad():
        for start_idx in range(0, m, output_chunk_size):
            end_idx = min(start_idx + output_chunk_size, m)

            X_chunk = torch.tensor(
                as_dense_f32(X[start_idx:end_idx, :]),
                dtype=torch.float32,
                device=device,
            )

            encoded_chunk = model.encoder(X_chunk)
            log_scale_chunk = model.scale_encoder(X_chunk).clamp(-10.0, 10.0)

            ll += -loss_fn_dense(model, X_chunk, constant_terms=True).item()

            Xnmf[start_idx:end_idx, :] = encoded_chunk.cpu().numpy()
            scales[start_idx:end_idx] = torch.exp(log_scale_chunk).squeeze(1).cpu().numpy()

    adata.obsm["X_nmf"] = Xnmf
    adata.obs["scale_nmf"] = scales
    adata.varm["V_nmf"] = model.v_norm().detach().cpu().numpy().T
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
