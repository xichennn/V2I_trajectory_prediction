from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from datamodules import V2XDataModule
from models.v2x_model import V2X    

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

if __name__ == '__main__':
    pl.seed_everything(2024)

    parser = ArgumentParser()
    parser.add_argument('--root', type=str, default='')
    parser.add_argument('--ckpt_path', type=str, default='')
    parser.add_argument('--train_batch_size', type=int, default=4)
    parser.add_argument('--val_batch_size', type=int, default=4)
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
    parser = V2X.add_model_specific_args(parser)
    args = parser.parse_args()
    datamodule = V2XDataModule(root=args.root,
                                    train_batch_size=args.train_batch_size,
                                    val_batch_size=args.val_batch_size,
                                    shuffle=args.shuffle,
                                    num_workers=args.num_workers,
                                    pin_memory=args.pin_memory,
                                    persistent_workers=args.persistent_workers
                                    )
    datamodule.setup()
    model_checkpoint = ModelCheckpoint(monitor=args.monitor, save_top_k=args.save_top_k, mode='min')
    trainer = pl.Trainer(max_epochs=args.max_epochs,
                        accelerator=args.accelerator,
                        devices=args.gpus,
                        gradient_clip_val=0.5,
                        accumulate_grad_batches=args.accu_batch,
                        callbacks=[model_checkpoint])
    model = V2X(**vars(args))
    if args.ckpt_path:
        model = V2X.load_from_checkpoint(checkpoint_path=args.ckpt_path, parallel=False)
    model.T_max = args.T_max
    model.max_epochs = args.max_epochs
    trainer.fit(model, datamodule)