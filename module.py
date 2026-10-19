import torch
import torch.nn as nn


class NormalDrift(nn.Module):
    """Normal drift A = Q B Q^T, split into dynamic planes and static lines.

    A is normal with singular values in [0, 1). B is block diagonal: first
    l = dim - p static 1x1 lines, each a copy scaled by lambda, then p // 2
    dynamic 2x2 planes, each a rotation by theta scaled by r. So p sets how
    much of the latent is treated as dynamic.

    self.training picks A or A*. Train mode keeps the lines and fits A in
    full; eval mode zeroes them, so .weight returns A*, the dynamics-only
    operator the cost measures along. A line can be predictable, lambda near
    1, and still be static, so the lines cannot be dropped by spectrum
    magnitude -- that is structural, which is why it happens here and not in
    the cost.
    """

    def __init__(self, dim, p=None, h=1):
        super().__init__()
        p = dim if p is None else p
        self.input_dim = dim
        self.h = h  # fit horizon (number of future lags)
        self.p = p  # dims spent on dynamic planes (p // 2 2x2 blocks)
        self.l = dim - p  # dims spent on static lines  # noqa: E741
        assert p % 2 == 0, f'p={p} must be even'
        assert self.l >= 0, f'p={p} must be <= dim={dim}'

        self.skew = nn.Parameter(torch.randn(dim, dim) * (dim**-0.5))
        self.r_ = nn.Parameter(torch.randn(p // 2))
        self.theta = nn.Parameter(torch.rand(p // 2) * torch.pi)
        self.lam_ = nn.Parameter(torch.randn(self.l))

    def orthogonal(self):
        """Q: the orthogonal frame, the eigenbasis of A A^T."""
        return torch.linalg.matrix_exp(self.skew - self.skew.T)

    def spectrum(self):
        """Line scales and plane radii, squashed into (-1, 1).

        Layout puts the l static lines first, then the p dynamic planes.
        """
        lam = self.lam_ / torch.sqrt(1.0 + self.lam_**2)
        r = self.r_.abs() / torch.sqrt(1.0 + self.r_**2)
        return lam, r, self.theta

    def block_diag(self, t=1):
        lam, r, theta = self.spectrum()
        d = self.input_dim
        B = self.skew.new_zeros(d, d)
        if self.training:
            line = torch.arange(self.l, device=B.device)
            B[line, line] = lam**t
        a = r**t * torch.cos(t * theta)
        b = r**t * torch.sin(t * theta)
        plane = self.l + torch.arange(self.p // 2, device=B.device) * 2
        B[plane, plane] = a
        B[plane + 1, plane + 1] = a
        B[plane, plane + 1] = -b
        B[plane + 1, plane] = b
        return B

    @property
    def weight(self):
        """A in train mode, A* (lines zeroed) in eval mode."""
        Q = self.orthogonal()
        return Q @ self.block_diag() @ Q.T

    def forward(self, x, c=None):
        # Predict lags 1..h from the anchor. Fitting always runs in train mode,
        # so the static lines are live.
        h = self.h
        Q = self.orthogonal()
        lam, r, theta = self.spectrum()
        k = torch.arange(1, h + 1, device=x.device, dtype=x.dtype)
        k = k.unsqueeze(-1)

        w = x @ Q
        out = w.new_empty(w.shape[:-2] + (h, self.input_dim))

        lines = w[..., : self.l]  # (..., 1, l)
        lam_k = lam**k  # (t, l)
        out[..., : self.l] = lam_k * lines

        pairs = w[..., self.l :].unflatten(-1, (self.p // 2, 2))
        w1, w2 = pairs.unbind(-1)
        r_k = r**k  # (t, p)
        cos = torch.cos(k * theta)
        sin = torch.sin(k * theta)
        y1 = r_k * (cos * w1 - sin * w2)
        y2 = r_k * (sin * w1 + cos * w2)
        out[..., self.l :] = torch.stack((y1, y2), dim=-1).flatten(-2)

        return out @ Q.T


class NonPSDNormalDrift(nn.Module):
    """NormalDrift without the squash: A = Q B Q^T with a raw spectrum.

    lambda and r may leave the unit interval, so Sigma = I - A A^T need not
    be PSD. Everything else matches NormalDrift, including A* at eval.
    lambda and r start at the squashed values NormalDrift would start from,
    so the two begin at the same A and differ only in where training may
    take the spectrum.
    """

    def __init__(self, dim, p=None, h=1):
        super().__init__()
        self.input_dim = dim
        self.h = h
        p = dim if p is None else p
        self.p = p
        self.l = dim - p  # noqa: E741
        assert p % 2 == 0, f'p={p} must be even'
        assert self.l >= 0, f'p={p} must be <= dim={dim}'

        self.skew = nn.Parameter(torch.randn(dim, dim) * (dim**-0.5))
        r = torch.randn(p // 2).abs()
        self.r_ = nn.Parameter(r / torch.sqrt(1.0 + r**2))
        self.theta = nn.Parameter(torch.rand(p // 2) * torch.pi)
        lam = torch.randn(self.l)
        self.lam_ = nn.Parameter(lam / torch.sqrt(1.0 + lam**2))

    def orthogonal(self):
        return torch.linalg.matrix_exp(self.skew - self.skew.T)

    def spectrum(self):
        return self.lam_, self.r_.abs(), self.theta

    def block_diag(self, t=1):
        lam, r, theta = self.spectrum()
        d = self.input_dim
        B = self.skew.new_zeros(d, d)
        if self.training:
            line = torch.arange(self.l, device=B.device)
            B[line, line] = lam**t
        a = r**t * torch.cos(t * theta)
        b = r**t * torch.sin(t * theta)
        plane = self.l + torch.arange(self.p // 2, device=B.device) * 2
        B[plane, plane] = a
        B[plane + 1, plane + 1] = a
        B[plane, plane + 1] = -b
        B[plane + 1, plane] = b
        return B

    @property
    def weight(self):
        Q = self.orthogonal()
        return Q @ self.block_diag() @ Q.T

    def forward(self, x, c=None):
        h = self.h
        Q = self.orthogonal()
        lam, r, theta = self.spectrum()
        k = torch.arange(1, h + 1, device=x.device, dtype=x.dtype)
        k = k.unsqueeze(-1)

        w = x @ Q
        out = w.new_empty(w.shape[:-2] + (h, self.input_dim))

        lines = w[..., : self.l]
        out[..., : self.l] = lam**k * lines

        pairs = w[..., self.l :].unflatten(-1, (self.p // 2, 2))
        w1, w2 = pairs.unbind(-1)
        r_k = r**k
        cos = torch.cos(k * theta)
        sin = torch.sin(k * theta)
        y1 = r_k * (cos * w1 - sin * w2)
        y2 = r_k * (sin * w1 + cos * w2)
        out[..., self.l :] = torch.stack((y1, y2), dim=-1).flatten(-2)

        return out @ Q.T


class NonNormalDrift(nn.Module):
    """Bounded but not normal: A = U diag(s) V^T, s in [0, 1), U != V.

    Sigma = I - A A^T = U (I - S^2) U^T is still PSD, but there is no shared
    eigenframe and no rotation planes.
    """

    def __init__(self, dim, h=1):
        super().__init__()
        self.input_dim = dim
        self.h = h
        self.skew_u = nn.Parameter(torch.randn(dim, dim) * (dim**-0.5))
        self.skew_v = nn.Parameter(torch.randn(dim, dim) * (dim**-0.5))
        self.s_ = nn.Parameter(torch.randn(dim))

    def spectrum(self):
        s = self.s_.abs()
        return s / torch.sqrt(1.0 + s**2)

    @property
    def weight(self):
        U = torch.linalg.matrix_exp(self.skew_u - self.skew_u.T)
        V = torch.linalg.matrix_exp(self.skew_v - self.skew_v.T)
        return (U * self.spectrum()) @ V.T

    def forward(self, x, c=None):
        A = self.weight
        d = A.shape[0]
        out = x.new_empty(x.shape[:-2] + (self.h, d))
        Ak = torch.eye(d, device=x.device, dtype=x.dtype)
        for t in range(self.h):
            Ak = Ak @ A
            out[..., t, :] = (x @ Ak.T).squeeze(-2)
        return out


class SymmetricDrift(nn.Module):
    """Bounded and normal without motion: A = Q diag(lambda) Q^T, with
    lambda in (-1, 1).

    Every block is a real 1x1 line, so there are no rotation planes. Unlike
    NormalDrift, the lines are the dynamics here, so eval mode keeps them.
    """

    def __init__(self, dim, h=1):
        super().__init__()
        self.input_dim = dim
        self.h = h
        self.skew = nn.Parameter(torch.randn(dim, dim) * (dim**-0.5))
        self.lam_ = nn.Parameter(torch.randn(dim))

    def orthogonal(self):
        return torch.linalg.matrix_exp(self.skew - self.skew.T)

    def spectrum(self):
        lam = self.lam_
        return lam / torch.sqrt(1.0 + lam**2)

    @property
    def weight(self):
        Q = self.orthogonal()
        return (Q * self.spectrum()) @ Q.T

    def forward(self, x, c=None):
        Q = self.orthogonal()
        lam = self.spectrum()
        k = torch.arange(1, self.h + 1, device=x.device, dtype=x.dtype)
        out = lam ** k.unsqueeze(-1) * (x @ Q)
        return out @ Q.T


class LinearDrift(nn.Module):
    """No structure: A = W, a free matrix."""

    def __init__(self, dim, h=1):
        super().__init__()
        self.input_dim = dim
        self.h = h
        self.W = nn.Parameter(torch.randn(dim, dim) * (dim**-0.5))

    @property
    def weight(self):
        return self.W

    def forward(self, x, c=None):
        A = self.weight
        d = A.shape[0]
        out = x.new_empty(x.shape[:-2] + (self.h, d))
        Ak = torch.eye(d, device=x.device, dtype=x.dtype)
        for t in range(self.h):
            Ak = Ak @ A
            out[..., t, :] = (x @ Ak.T).squeeze(-2)
        return out


class PCABasis(nn.Module):
    """Centered covariance C of the frozen encoder's embeddings.

    Stored as ``weight`` so the planning costs read it like a drift
    checkpoint:
    C is symmetric PSD, so its left singular vectors are the principal
    directions, ordered by variance, so LP2GoalMSE's top-k basis is the
    top-k PCA subspace -- the same cost code as ours, which is what makes
    it a control rather than a reimplementation.
    """

    def __init__(self, dim):
        super().__init__()
        self.input_dim = dim
        self.register_buffer('weight', torch.zeros(dim, dim))


class QuasimetricHead(nn.Module):
    """Asymmetric latent distance d(z_s, z_g) calibrated to temporal
    reachability.

    TD-JEPA's head (module.py @ b4c17ca), logic unchanged. Metric-Residual-
    Network style: a shared projection splits into a symmetric Euclidean
    component and an asymmetric max-ReLU component. The result is non-negative
    and asymmetric in the (state, goal) order, so it can capture directed
    steps-to-goal (control reachability is not symmetric). Trained to regress
    the temporal gap between a state and a future state.
    """

    def __init__(self, input_dim, hidden_dim=512, sym_dim=128, asym_dim=128):
        super().__init__()
        self.sym_dim = sym_dim
        self.asym_dim = asym_dim
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, sym_dim + asym_dim),
        )

    def project(self, z):
        feats = self.backbone(z)
        return feats[..., : self.sym_dim], feats[..., self.sym_dim :]

    def forward_parts(self, z_s, z_g):
        """Return symmetric and asymmetric distance components."""
        sym_s, asym_s = self.project(z_s)
        sym_g, asym_g = self.project(z_g)
        sym = (sym_s - sym_g).pow(2).sum(dim=-1).clamp_min(1e-12).sqrt()
        asym = torch.relu(asym_s - asym_g).amax(dim=-1)
        return sym, asym

    def forward(self, z_s, z_g):
        """Return directed distance d(z_s -> z_g), shape ``z_s.shape[:-1]``."""
        sym, asym = self.forward_parts(z_s, z_g)
        return sym + asym
