"""Planning costs."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_worldmodel.wm.utils import load_pretrained


class LP2GoalMSE(nn.Module):
    """Goal MSE after projecting onto the top-k directions of A.

        cost = || U_k^T (z_T - z_goal) ||^2

    U_k is the top-k left singular vectors of A, which are the eigenvectors
    of A A^T, ordered by singular value -- from least to most transition
    noise, 1 - s_i^2. Taking k = n is a norm-preserving change of basis and
    recovers the plain goal MSE exactly.
    """

    def __init__(
        self,
        drift_ckpt: str,
        k: int,
        pred_key: str = 'predicted_emb',
        goal_key: str = 'goal_emb',
    ) -> None:
        super().__init__()
        self.k = k
        self.pred_key = pred_key
        self.goal_key = goal_key

        # eval() so a NormalDrift returns A* (static lines zeroed).
        A = load_pretrained(drift_ckpt).eval().weight.detach().float()
        U, _, _ = torch.linalg.svd(A)
        self.register_buffer('basis', U[:, :k].contiguous())  # (n, k)

    def forward(self, info_dict: dict) -> torch.Tensor:
        pred_emb = info_dict[self.pred_key]
        goal_emb = info_dict[self.goal_key][:, None, -1:, :].expand_as(pred_emb)
        basis = self.basis.to(device=pred_emb.device, dtype=pred_emb.dtype)

        pred = pred_emb[..., -1, :] @ basis
        goal = goal_emb[..., -1, :].detach() @ basis
        return F.mse_loss(pred, goal, reduction='none').sum(dim=-1)


class RandomGoalMSE(nn.Module):
    """Control for LP2GoalMSE: the same cost on a random basis.

    U_k is k columns of a Haar-random orthogonal matrix drawn from seed, so
    any gain that belongs to the fitted basis shows up as a gap between the
    two. Taking k = n recovers the plain goal MSE for any seed.
    """

    def __init__(
        self,
        dim: int,
        k: int,
        seed: int = 0,
        pred_key: str = 'predicted_emb',
        goal_key: str = 'goal_emb',
    ) -> None:
        super().__init__()
        self.k = k
        self.seed = seed
        self.pred_key = pred_key
        self.goal_key = goal_key

        gen = torch.Generator().manual_seed(seed)
        M = torch.randn(dim, dim, generator=gen)
        Q, _ = torch.linalg.qr(M)
        self.register_buffer('basis', Q[:, :k].contiguous())  # (n, k)

    def forward(self, info_dict: dict) -> torch.Tensor:
        pred_emb = info_dict[self.pred_key]
        goal_emb = info_dict[self.goal_key][:, None, -1:, :].expand_as(
            pred_emb
        )
        basis = self.basis.to(device=pred_emb.device, dtype=pred_emb.dtype)

        pred = pred_emb[..., -1, :] @ basis
        goal = goal_emb[..., -1, :].detach() @ basis
        return F.mse_loss(pred, goal, reduction='none').sum(dim=-1)


class TemporalDistance(nn.Module):
    """TD-JEPA (frozen): blend of a learned directed distance and latent l^2
    at the terminal frame,

        cost = (1 - a) d(z_T -> z_g) + a ||z_T - z_g||^2 ,

    where ``d`` is a ``QuasimetricHead`` fit by ``fit_td.py``. The ``td_jepa``
    branch of ``JEPA.criterion`` (TD-JEPA jepa.py @ b4c17ca) with
    ``terminal_weight = 1``: ``a = 0`` is pure ``d`` and ``a = 1`` is the plain
    goal MSE. The terms are not rescaled (``d`` is in window steps), matching
    their ``mse_blend``. ``d`` is asymmetric, so the argument order is state
    first, goal second.
    """

    def __init__(
        self,
        td_ckpt: str,
        alpha: float = 0.1,
        pred_key: str = 'predicted_emb',
        goal_key: str = 'goal_emb',
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.pred_key = pred_key
        self.goal_key = goal_key
        self.td = load_pretrained(td_ckpt).eval().requires_grad_(False)

    def forward(self, info_dict: dict) -> torch.Tensor:
        pred = info_dict[self.pred_key][..., -1, :]  # (B, S, D)
        goal = info_dict[self.goal_key][:, None, -1, :].detach()
        goal = goal.expand_as(pred)
        self.td.to(device=pred.device, dtype=pred.dtype)

        cost = self.td(pred, goal)
        if self.alpha > 0.0:
            mse = (pred - goal).pow(2).sum(dim=-1)
            cost = (1.0 - self.alpha) * cost + self.alpha * mse
        return cost
