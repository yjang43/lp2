"""Fit TD-JEPA's quasimetric head on a frozen world model's embeddings.

The learned-distance baseline: d(z_s -> z_g) is regressed to the number of
steps between the two frames, so the planner can rank candidates by
steps-to-goal rather than by latent distance. Only the head trains; the
encoder and predictor stay frozen, exactly as for the drift.
"""

import os
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
import torch.nn.functional as F
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf

from fit import SaveCkptCallback, embedding_cache, load_frozen_lewm


def temporal_distance_loss(head, emb):
    """Regress d(z_i -> z_j) to the gap j - i, with cross-window negatives.

    One (i, j) pair per row, j > i. A goal taken from another row of the
    batch is a heuristic negative whose distance is pushed past the window;
    that hinge is what makes the distance monotonic in steps-to-goal rather
    than merely small on positives.

    TD-JEPA's canonical setting (config/train/variant/td_jepa.yaml): raw step
    targets, hinge margin T - 1 at unit weight, no auxiliary terms.
    """
    batch, seq_len, _ = emb.shape
    device = emb.device
    rows = torch.arange(batch, device=device)
    i = torch.randint(0, seq_len - 1, (batch,), device=device)
    offset = torch.randint(1, seq_len, (batch,), device=device)
    j = torch.minimum(i + offset, torch.full_like(i, seq_len - 1))
    target = (j - i).float()

    z_s, z_g = emb[rows, i], emb[rows, j]
    d_pos = head(z_s, z_g)
    pos_loss = F.smooth_l1_loss(d_pos, target)

    perm = torch.randperm(batch, device=device)
    clash = perm == rows
    perm[clash] = (perm[clash] + 1) % batch
    d_neg = head(z_s, emb[perm, j])
    neg_loss = torch.relu(float(seq_len - 1) - d_neg).mean()

    metrics = {
        'td_pos_loss': pos_loss,
        'td_neg_loss': neg_loss,
        'td_pred_dist_mean': d_pos.mean(),
        'td_neg_dist_mean': d_neg.mean(),
        'td_target_mean': target.mean(),
    }
    return pos_loss + neg_loss, metrics


def td_forward(self, batch, stage):
    emb = batch['emb'].float()  # (B, T, D), cached
    loss, metrics = temporal_distance_loss(self.model.td, emb)

    output = {'td_loss': loss, 'loss': loss, **metrics}
    self.log_dict(
        {f'{stage}/{k}': v.detach() for k, v in output.items()},
        on_step=True,
        sync_dist=True,
    )
    return output


@hydra.main(
    version_base=None, config_path='./config/train', config_name='td'
)
def run(cfg):
    cache_dir = os.environ.get('LOCAL_DATASET_DIR', None)
    cache_name = embedding_cache(cfg)

    # The hinge margin is the window minus one, so the window length is part
    # of the method, not a free knob.
    num_steps = cfg.wm.history_size + cfg.wm.num_preds
    assert num_steps == 8, f'TD-JEPA window is 8 frames, got {num_steps}'

    print(
        f'Loading embedding cache "{cache_name}" '
        f'(num_steps={num_steps}, frameskip={cfg.wm.frameskip})'
    )
    dataset = swm.data.load_dataset(
        cache_name,
        transform=None,
        cache_dir=cache_dir,
        num_steps=num_steps,
        frameskip=cfg.wm.frameskip,
        keys_to_load=['emb'],
        keys_to_cache=['emb'],
    )

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset,
        lengths=[cfg.train_split, 1 - cfg.train_split],
        generator=rnd_gen,
    )
    train = torch.utils.data.DataLoader(
        train_set, **cfg.loader, generator=rnd_gen
    )
    val_cfg = {**cfg.loader}
    val_cfg['shuffle'] = False
    val_cfg['drop_last'] = False
    val = torch.utils.data.DataLoader(val_set, **val_cfg)

    world_model = load_frozen_lewm(cfg.pretrained_ckpt)
    world_model.td = hydra.utils.instantiate(cfg.td)

    optimizers = {
        'td_opt': {
            'modules': 'model.td',  # only the head is optimised
            'optimizer': dict(cfg.optimizer),
            'scheduler': {'type': 'LinearWarmupCosineAnnealingLR'},
            'interval': 'epoch',
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model=world_model,
        forward=td_forward,
        optim=optimizers,
    )

    run_id = cfg.get('subdir') or ''
    run_dir = Path(
        swm.data.utils.get_cache_dir(sub_folder='checkpoints'), run_id
    )

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / 'config.yaml', 'w') as f:
        OmegaConf.save(cfg, f)

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[
            SaveCkptCallback(run_name=cfg.output_model_name, cfg=cfg.td, attr='td')
        ],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    ckpt_path = run_dir / f'{cfg.output_model_name}_weights.ckpt'
    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=ckpt_path if ckpt_path.exists() else None,
    )
    manager()


if __name__ == '__main__':
    run()
