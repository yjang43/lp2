"""Fit a PCA basis on a frozen world model's embeddings.

Nothing is trained: the basis is the centred covariance of the cached
embeddings, which LP2GoalMSE reads exactly as it reads a drift.
"""

import os

import hydra
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from stable_worldmodel.wm.utils import save_pretrained

from fit import embedding_cache
from module import PCABasis


def fit_covariance(loader):
    """Centred covariance of every embedding a loader yields, in float64.

    Streams sums, so the frames never sit in memory together:
    C = (sum z zT - n mu muT) / (n - 1).
    """
    n, total, outer = 0, None, None
    for batch in loader:
        z = batch['emb'].reshape(-1, batch['emb'].shape[-1]).double()
        if total is None:
            total = z.new_zeros(z.shape[1])
            outer = z.new_zeros(z.shape[1], z.shape[1])
        n += z.shape[0]
        total += z.sum(dim=0)
        outer += z.T @ z
    mean = total / n
    return (outer - n * torch.outer(mean, mean)) / (n - 1)


@hydra.main(version_base=None, config_path='./config/train', config_name='pca')
def run(cfg):
    cache_dir = os.environ.get('LOCAL_DATASET_DIR', None)
    cache_name = embedding_cache(cfg)

    print(f'Loading embedding cache "{cache_name}" (single frames)')
    dataset = swm.data.load_dataset(
        cache_name,
        transform=None,
        cache_dir=cache_dir,
        num_steps=1,
        frameskip=1,
        keys_to_load=['emb'],
        keys_to_cache=['emb'],
    )

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, _ = spt.data.random_split(
        dataset,
        lengths=[cfg.train_split, 1 - cfg.train_split],
        generator=rnd_gen,
    )

    loader_cfg = {**cfg.loader, 'shuffle': False, 'drop_last': False}
    train = torch.utils.data.DataLoader(train_set, **loader_cfg)

    cov = fit_covariance(train)
    basis = PCABasis(cov.shape[0])
    basis.weight.copy_(cov.float())

    save_pretrained(
        basis,
        run_name=cfg.output_model_name,
        config={'_target_': 'module.PCABasis', 'dim': cfg.embed_dim},
        filename='pca.pt',
    )
    print(f'wrote {cfg.output_model_name}/pca.pt')


if __name__ == '__main__':
    run()
