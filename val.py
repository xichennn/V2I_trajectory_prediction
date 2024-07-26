from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch_geometric.loader import DataLoader

from datasets import V2XDataset
from models.v2x_model import V2X    

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


if __name__ == '__main__':
    pl.seed_everything(2024)

    parser = ArgumentParser()
    parser.add_argument('--root', type=str, default='')
    parser.add_argument('--ckpt_path', type=str, default='')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--persistent_workers', type=bool, default=True)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--max_epochs', type=int, default=64)
    parser.add_argument('--monitor', type=str, default='val_minFDE', choices=['val_minADE', 'val_minFDE', 'val_minMR'])
    parser.add_argument('--save_top_k', type=int, default=6)
    parser.add_argument('--accelerator', type=str, default="cpu")
    parser.add_argument('--accu_batch', type=int, default=4)
    parser.add_argument('--check_val_every_n_epoch', type=int, default=5)
    parser.add_argument('--model', type=str, default='V2X')
    parser = V2X.add_model_specific_args(parser)
    args = parser.parse_args()

    trainer = pl.Trainer()
    if args.model == 'V2X':
        model = V2X.load_from_checkpoint(checkpoint_path=args.ckpt_path, parallel=False)
        val_dataset = V2XDataset(root=args.root, split='val', local_radius=model.hparams.local_radius)

    val = list(range(0, len(val_dataset)))
    val_set = torch.utils.data.Subset(val_dataset, val)
    dataloader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                            pin_memory=args.pin_memory, persistent_workers=args.persistent_workers)
    trainer.validate(model, dataloader)
