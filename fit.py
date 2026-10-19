"""Fit the latent state dynamics A on a frozen, pretrained world model."""

import os
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf
from stable_pretraining import data as dt
from stable_worldmodel.data.formats.hdf5 import HDF5Writer
from stable_worldmodel.wm.utils import load_pretrained, save_pretrained


def get_img_preprocessor(source: str, target: str, img_size: int = 224):
    imagenet_stats = dt.dataset_stats.ImageNet
    to_image = dt.transforms.ToImage(
        **imagenet_stats, source=source, target=target
    )
    resize = dt.transforms.Resize(img_size, source=source, target=target)
    return dt.transforms.Compose(to_image, resize)


class SaveCkptCallback(Callback):
    """Save ``model.<attr>`` every ``epoch_interval`` epochs."""

    def __init__(self, run_name, cfg, attr='drift', epoch_interval: int = 1):
        super().__init__()
        self.run_name = run_name
        self.cfg = cfg
        self.attr = attr
        self.epoch_interval = epoch_interval

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)
        epoch = trainer.current_epoch + 1
        if trainer.is_global_zero and (
            epoch % self.epoch_interval == 0 or epoch == trainer.max_epochs
        ):
            save_pretrained(
                getattr(pl_module.model, self.attr),
                run_name=self.run_name,
                config=self.cfg,
                filename=f'{self.attr}_epoch_{epoch}.pt',
            )


def load_frozen_lewm(ckpt_name):
    """Load the pretrained LeWM frozen.

    eval() matters: the projector's BatchNorm both updates running stats
    and rejects batch size 1 in train mode.
    """
    model = load_pretrained(ckpt_name)
    model.requires_grad_(False)
    model.eval()
    return model


def build_cache(cfg, dataset_name, cache_dir, cache_path):
    """Encode every frame once through the frozen encoder into an HDF5 cache.

    Frames are flattened across episodes into large encoder batches (episodes
    can be shorter than a useful batch) and regrouped by episode length.
    """
    source = swm.data.load_dataset(
        dataset_name, cache_dir=cache_dir, keys_to_load=['pixels']
    )
    model = load_frozen_lewm(cfg.pretrained_ckpt).to('cuda')
    transform = get_img_preprocessor('pixels', 'pixels', cfg.img_size)
    n_ep = len(source.lengths)
    print(f'[cache] building {cache_path} from {n_ep} episodes')

    with HDF5Writer(cache_path, mode='overwrite') as writer:
        buf, ep_lens = [], []
        for ep in range(n_ep):
            end = int(source.lengths[ep])
            raw = source._load_slice(ep, 0, end)['pixels']
            buf.append(transform({'pixels': raw})['pixels'])
            ep_lens.append(buf[-1].size(0))

            if sum(ep_lens) >= 512 or ep == n_ep - 1:
                pixels = torch.cat(buf)[None].to('cuda')
                with torch.no_grad():
                    emb = model.encode({'pixels': pixels})['emb'][0]
                emb = emb.half().cpu().numpy()
                offset = 0
                for length in ep_lens:
                    writer.write_episode(
                        {'emb': emb[offset : offset + length]}
                    )
                    offset += length
                buf, ep_lens = [], []
                print(f'[cache] {ep + 1}/{n_ep} episodes', flush=True)


def embedding_cache(cfg):
    """Name of the frozen encoder's embedding cache, building it if absent.

    Keyed by checkpoint as well as dataset, so fitting on two world models
    over the same pixels never reuses the wrong embeddings. Shared with
    fit_pca.py and fit_td.py so all three read one cache.
    """
    dataset_name = cfg.data.dataset.name
    cache_dir = os.environ.get('LOCAL_DATASET_DIR', None)
    wm_tag = str(Path(cfg.pretrained_ckpt).with_suffix('')).replace('/', '_')
    cache_name = f'{Path(dataset_name).stem}.{wm_tag}.cache.h5'
    datasets_dir = Path(swm.data.utils.get_cache_dir(cache_dir, 'datasets'))
    cache_path = datasets_dir / cache_name
    if not cache_path.exists():
        build_cache(cfg, dataset_name, cache_dir, cache_path)
    return cache_name


def drift_forward(self, batch, stage, cfg):
    """Mean over lags t = 1..h of ||A^t z_0 - z_t||^2, on cached embeddings.

    The anchor predicts each lag through the closed-form power, so every lag
    constrains the same A.
    """
    ctx_len = cfg.wm.history_size
    emb = batch['emb'].float()  # (B, num_steps, n)

    ctx_emb = emb[:, :ctx_len]
    tgt_emb = emb[:, ctx_len:]
    pred_emb = self.model.drift(ctx_emb)  # (B, h, n)
    tgt_emb = tgt_emb[:, : pred_emb.shape[1]]

    output = {'drift_loss': (pred_emb - tgt_emb).pow(2).mean()}
    output['loss'] = output['drift_loss']

    losses_dict = {
        f'{stage}/{k}': v.detach() for k, v in output.items() if 'loss' in k
    }
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output


@hydra.main(version_base=None, config_path='./config/train', config_name='normal_drift')
def run(cfg):
    cache_dir = os.environ.get('LOCAL_DATASET_DIR', None)
    cache_name = embedding_cache(cfg)

    # One anchor frame plus num_preds futures, strided by frameskip: lag t in
    # the loss is t strided steps from the anchor.
    frameskip = cfg.wm.frameskip
    num_steps = cfg.wm.history_size + cfg.wm.num_preds
    print(
        f'Loading embedding cache "{cache_name}" '
        f'(num_steps={num_steps}, frameskip={frameskip})'
    )
    dataset = swm.data.load_dataset(
        cache_name,
        transform=None,
        cache_dir=cache_dir,
        num_steps=num_steps,
        frameskip=frameskip,
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
    world_model.drift = hydra.utils.instantiate(cfg.drift)

    total_steps = cfg.trainer.max_epochs * len(train)
    optimizers = {
        'drift_opt': {
            'modules': 'model.drift',  # only the drift is optimised
            'optimizer': dict(cfg.optimizer),
            'scheduler': {
                'type': 'LinearWarmupCosineAnnealingLR',
                'warmup_steps': max(1, int(0.01 * total_steps)),
                'max_steps': total_steps,
            },
            'interval': 'epoch',
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model=world_model,
        forward=partial(drift_forward, cfg=cfg),
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
            SaveCkptCallback(run_name=cfg.output_model_name, cfg=cfg.drift)
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
