from argparse import ArgumentParser
import sys, os
sys.path.append('/Users/xichen/Documents/paper2-traj-pred/DAIR-V2X-Seq/projects/HiVT_plugin')

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

# from datamodules import TFDDataModule
# from models.hivt import HiVT
from coop_models.check_coop_model import Coop

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

# %%
from typing import Callable, Optional

from pytorch_lightning import LightningDataModule
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset

from dataloader.v2x_dataset import V2XDataset


class V2XDataModule(LightningDataModule):

    def __init__(self,
                 root: str,
                 train_batch_size: int,
                 val_batch_size: int,
                 shuffle: bool = True,
                 num_workers: int = 8,
                 pin_memory: bool = True,
                 persistent_workers: bool = True,
                 train_transform: Optional[Callable] = None,
                 val_transform: Optional[Callable] = None,
                 local_radius: float = 50) -> None:
        super(V2XDataModule, self).__init__()
        self.root = root
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.num_workers = num_workers
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.local_radius = local_radius

    def prepare_data(self) -> None:
        V2XDataset(self.root, 'val', transform=self.val_transform, local_radius=self.local_radius)
        V2XDataset(self.root, 'train', transform=self.train_transform, local_radius=self.local_radius)

    def setup(self, stage: Optional[str] = None) -> None:
        self.val_dataset = V2XDataset(self.root, 'val', self.val_transform, self.local_radius)
        self.train_dataset = V2XDataset(self.root, 'train', self.train_transform, self.local_radius)

    def train_dataloader(self):
        return DataLoader(Subset(self.train_dataset, range(len(self.train_dataset)//2)), batch_size=self.train_batch_size, shuffle=self.shuffle,
                          num_workers=self.num_workers, pin_memory=self.pin_memory,
                          persistent_workers=self.persistent_workers)
    # Subset(self.train_dataset, range(len(self.train_dataset)//2))
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.val_batch_size, shuffle=False, num_workers=self.num_workers,
                          pin_memory=self.pin_memory, persistent_workers=self.persistent_workers)


# %%

if __name__ == '__main__':
    pl.seed_everything(2022)

    parser = ArgumentParser()
    parser.add_argument('--root', type=str, default='/Users/xichen/Documents/paper2-traj-pred/DAIR-V2X-Seq/dataset/v2x-seq-tfd/V2X-Seq-TFD-Example/cooperative-vehicle-infrastructure/vehicle-trajectories/')
    parser.add_argument('--ckpt_path', type=str, default='')
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--val_batch_size', type=int, default=32)
    parser.add_argument('--shuffle', type=bool, default=True)
    # parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--persistent_workers', type=bool, default=True)
    # parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--gpus', type=int, default=0)
    parser.add_argument('--max_epochs', type=int, default=64)
    parser.add_argument('--monitor', type=str, default='val_minFDE', choices=['val_minADE', 'val_minFDE', 'val_minMR'])
    parser.add_argument('--save_top_k', type=int, default=5)
    parser.add_argument('--check_val_every_n_epoch', type=int, default=5)
    
    parser = Coop.add_model_specific_args(parser)
    args = parser.parse_args()
    datamodule = V2XDataModule.from_argparse_args(args)
    # datamodule = V2XDataModule(root = args.root, 
    #                            train_batch_size=args.train_batch_size,
    #                            val_batch_size=args.val_batch_size,
    #                            shuffle=args.shuffle,
    #                            num_workers=args.num_workers,
    #                            pin_memory=args.pin_memory,
    #                            persistent_workers=args.persistent_workers,
    #                            local_radius=args.model_radius)
    datamodule.setup()
    model_checkpoint = ModelCheckpoint(monitor=args.monitor, save_top_k=args.save_top_k, mode='min')
    trainer = pl.Trainer.from_argparse_args(args, accelerator='cpu', callbacks=[model_checkpoint])
    model = Coop(**vars(args))
    if args.ckpt_path:
        model = model.load_from_checkpoint(checkpoint_path=args.ckpt_path, parallel=False)
    model.T_max = args.T_max
    model.max_epochs = args.max_epochs
    trainer.fit(model, datamodule)