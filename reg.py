"""The latent regularizers train.py swaps between, as published.

Every regularizer is handed the latent as ``(T, B, D)`` and returns one
scalar -- the convention stable-worldmodel's SIGReg already follows, so that
arm needs no code here and is selected straight from the config.

These are verbatim ports but for that convention: VISReg returns the sum of
its three terms rather than the terms, and RDMReg transposes back, since it
matches per timestep with the batch as the sample axis.

Strength is not a property of a regularizer: ``reg_weight`` scales whichever
one is selected. Leaving ``reg`` unset is the stop-gradient arm.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn


class VISReg(nn.Module):
    """Sliced-Wasserstein isotropic Gaussian regularizer.

    Ported from https://github.com/HaiyuWu/visreg (arXiv:2606.02572); kept
    local to this arm so the swm package stays untouched.

    Scale and shape are fit separately: shape is measured on
    ``z / std.detach()``, so it is blind to scale by construction, and
    ``(std - 1)^2`` keeps pulling as the embedding collapses. SIGReg's
    Epps-Pulley statistic flattens in that regime -- its gradient vanishes
    exactly when anti-collapse pressure is most needed, which is the
    difference this arm exists to test.
    """

    def __init__(
        self,
        num_proj=1024,
        scale_weight=1.0,
        shape_weight=1.0,
        center_weight=1.0,
    ):
        super().__init__()
        self.num_proj = num_proj
        self.scale_weight = scale_weight
        self.shape_weight = shape_weight
        self.center_weight = center_weight
        self.target_key = None
        self.target = None

    def gaussian_quantiles(self, n, device, dtype):
        """Inverse-CDF knots the sorted projections are matched against.

        Keyed on dtype as well as size: the trainer runs bf16, and a cached
        fp32 target would silently mismatch once ``drop_last`` pins n.
        """
        key = (n, device, dtype)
        if self.target_key != key:
            q = torch.arange(1, n + 1, device=device, dtype=dtype)
            q = q.div_(n + 1)
            self.target = torch.erfinv(2 * q - 1).mul_(math.sqrt(2))
            self.target_key = key
        return self.target

    def forward(self, proj):
        """
        proj: (T, B, D)
        """
        B, D = proj.size(-2), proj.size(-1)

        mu = proj.mean(dim=-2, keepdim=True)
        center_loss = mu.pow(2).mean()

        centered = proj - mu
        std = centered.norm(dim=-2).div(math.sqrt(B)).clamp_min(1e-6)
        scale_loss = (std - 1.0).pow(2).mean()

        normed = centered / std.detach().unsqueeze(-2)
        A = torch.randn(D, self.num_proj, device=proj.device, dtype=proj.dtype)
        A = F.normalize(A, dim=0)
        sorted_proj = (normed @ A).sort(dim=-2).values
        target = self.gaussian_quantiles(
            B, sorted_proj.device, sorted_proj.dtype
        ).view(1, B, 1)
        shape_loss = (sorted_proj - target).pow(2).mean()

        return (
            self.scale_weight * scale_loss
            + self.shape_weight * shape_loss
            + self.center_weight * center_loss
        )


class RepReLU(nn.Module):
    """LpWM's link (arXiv:2608.22764, Sec. 2.2): forward ReLU(x), backward
    through GELU(x), so the code has exact zeros without the dying-ReLU
    gradient. Appended to both MLP heads by the config; named there as
    reg.RepReLU, so this module must stay importable at eval and fit time
    as well as during training."""

    def forward(self, x):
        gelu_x = F.gelu(x)
        relu_x = F.relu(x)
        return gelu_x - gelu_x.detach() + relu_x.detach()


class RDMReg(nn.Module):
    """Rectified Distribution Matching Regularization (LpWM).

    Two-sample sliced W2 between the latent and a fresh sample of the same
    shape from link(mu + sigma * GN_p). Ported from lpworldmodel
    (lpwm_swm/loss.py, Kuang et al., arXiv:2608.22764) without the RMS
    normalisation option, which the paper's stable-worldmodel runs leave
    off. The paper's setting is link=ReLU, p=1, mu=0, a rectified product
    Laplace: the code is non-negative and exactly sparse, provided the model
    heads carry the same link (see the config). link=Identity with p=2 is
    the paper's dense LeWM control.
    """

    def __init__(
        self, p=1.0, mu=0.0, link_function_type='ReLU', num_slices=1024
    ):
        super().__init__()
        self.p = float(p)
        self.mu = mu
        self.num_slices = num_slices
        # sigma such that GN_p(0, sigma) has unit variance
        self.sigma = math.sqrt(
            math.gamma(1 / self.p) / math.gamma(3 / self.p)
        ) / self.p ** (1 / self.p)
        if link_function_type == 'ReLU':
            self.link_function = nn.ReLU()
        elif link_function_type == 'Identity':
            self.link_function = nn.Identity()
        else:
            raise ValueError(f'Unsupported link: {link_function_type}')

    def sample_gn(self, shape, device):
        """Standard GN_p before the sigma rescale: Laplace(0,1) at p=1,
        N(0,1) at p=2, Gamma-based otherwise."""
        p = self.p
        if p == 1.0:
            lap = torch.distributions.Laplace(
                torch.tensor(0.0, device=device),
                torch.tensor(1.0, device=device),
            )
            return lap.sample(shape)
        if p == 2.0:
            return torch.randn(shape, device=device)
        sign = torch.empty(shape, device=device).bernoulli_(0.5) * 2 - 1
        g = torch.distributions.Gamma(1.0 / p, 1.0).sample(shape).to(device)
        return sign * (p * g).pow(1.0 / p)

    def sample_target(self, shape, device):
        return self.link_function(
            self.mu + self.sigma * self.sample_gn(shape, device)
        )

    def forward(self, z):
        """z: (T, B, D); the batch is the sample axis sorted along."""
        z = z.transpose(0, 1)
        D = z.shape[-1]
        target = self.sample_target(tuple(z.shape), z.device).to(z.dtype)

        A = torch.randn(self.num_slices, D, device=z.device, dtype=z.dtype)
        A = A / A.norm(dim=1, keepdim=True)

        proj_z = (z @ A.T).sort(dim=0).values
        proj_t = (target @ A.T).sort(dim=0).values
        return (proj_z - proj_t).pow(2).mean()
