from argparse import ArgumentParser

import sys
import os
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.dirname(SCRIPT_DIR))
sys.path.append('/Users/xichen/Documents/paper2-traj-pred/DAIR-V2X-Seq/projects/HiVT_plugin')
# from dataloader.temporal_data import TFDDataset
from dataloader.v2x_dataset import V2XDataset
from torch_geometric.loader import DataLoader

from Coop_plugin.coop_models.coop_model import CoopModel
# from models.hivt import HiVT

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from torch.optim import Adam, AdamW
from utils import ScheduledOptim
from Coop_plugin.metrics.loss import NLLloss
import math
from tqdm import tqdm

infra_traj_dir = '/Users/xichen/Documents/paper2-traj-pred/DAIR-V2X-Seq/dataset/v2x-seq-tfd/V2X-Seq-TFD-Example/cooperative-vehicle-infrastructure/infrastructure-trajectories/'
car_traj_dir = '/Users/xichen/Documents/paper2-traj-pred/DAIR-V2X-Seq/dataset/v2x-seq-tfd/V2X-Seq-TFD-Example/cooperative-vehicle-infrastructure/vehicle-trajectories/'

train_dataset = V2XDataset(car_traj_dir, 'train', transform=None, local_radius=50)
val_dataset = V2XDataset(car_traj_dir, 'val', transform=None, local_radius=50)

# infra_val_dataset = TFDDataset(infra_traj_dir, 'val', transform=None, local_radius=50)
# infra_train_dataset = TFDDataset(infra_traj_dir, 'train', transform=None, local_radius=50)

# car_val_dataset = TFDDataset(car_traj_dir, 'val', transform=None, local_radius=50)
# car_train_dataset = TFDDataset(car_traj_dir, 'train', transform=None, local_radius=50)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = CoopModel(device=device)



# pred = net(train_dataset[0])
#args
batch_size = 2
num_workers = 4
lr = 1e-3
betas = (0.9, 0.999)
weight_decay = 0.0001
warmup_epoch = 10
lr_update_freq = 10
lr_decay_rate = 0.9

log_freq = 10
save_folder = ""
model_path = "/home/u6/xic/v2x_projects"
ckpt_path = None
verbose = True

train_loader = DataLoader(train_dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        pin_memory= True,
                        shuffle=True,
                        persistent_workers=True)
val_loader = DataLoader(val_dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        pin_memory= True,
                        shuffle=True,
                        persistent_workers=True)
# pred = net(next(iter(train_loader)))

#loss
criterion = NLLloss(alpha=0.5, use_variance=False, device=device)

# init optimizer
optim = AdamW(net.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
optm_schedule = ScheduledOptim(
    optim,
    lr,
    n_warmup_epoch=warmup_epoch,
    update_rate=lr_update_freq,
    decay_rate=lr_decay_rate
)

net = net.to(device)

# iteration
training = net.training
avg_loss = 0.0
avg_loss_val = 0.0
losses_train =[]
losses_val = []

epochs = 100
minVal = math.inf

for epoch in range(epochs):
    avg_loss =0.0
    ## Training:_______________________________________________________________
    training = True
    data_iter = tqdm(
    enumerate(train_loader),
    desc="{}_Ep_{}: loss: {:.5e}; avg_loss: {:.5e}".format("train" if training else "eval",
                                                            epoch,
                                                            0.0,
                                                            avg_loss),
    total=len(train_loader),
    bar_format="{l_bar}{r_bar}"
)
    count = 0

    for i, data in data_iter: #next(iter(train_loader))
        data = data.to(device)

        if training:
            optm_schedule.zero_grad()
            pred = net(data)
            loss = criterion(pred['traj'][:,:,:,:2], data.y, pred['log_probs'])
            loss.backward()
            losses_train.append(loss.detach().item())

            torch.nn.utils.clip_grad_norm_(net.parameters(), 100)
            optim.step()
            avg_loss += loss.detach().item()
            count += 1

            # print log info
            desc_str = "[Info: Device_{}: {}_Ep_{}: loss: {:.5e}; avg_loss: {:.5e}]".format(
                0,
                "train" if training else "eval",
                epoch,
                loss.item(),
                avg_loss / count)
            data_iter.set_description(desc=desc_str, refresh=True)
            if training:
                learning_rate = optm_schedule.step_and_update_lr()
            if epoch%10==0:
                print("learning_rate: ", learning_rate)
            
    ## Val:_______________________________________________________________________________________________________________________________
    training = False
    # model.eval()
    avg_loss_val = 0.0
    count_val = 0
    data_iter_val = tqdm(enumerate(val_loader), desc="{}_Ep_{}: loss: {:.5e}; avg_loss: {:.5e}".format("eval",
                        epoch,
                        0.0,
                        avg_loss_val),
                        total=len(val_loader),
                        bar_format="{l_bar}{r_bar}"
                        )
    for i, data_val in data_iter_val:
        data_val = data_val.to(device)

        with torch.no_grad():
            pred_val = net(data_val)
            loss_val = criterion(pred_val['traj'][:,:,:,:2],
                                 data_val.y, pred_val['log_probs'])

        losses_val.append(loss_val.detach().item())
        avg_loss_val += loss_val.detach().item()
        count_val += 1

        # print log info
        desc_str_val = "[Info: Device_{}: {}_Ep_{}: loss: {:.5e}; avg_loss: {:.5e}]".format(
            0,
            "eval",
            epoch,
            loss_val.item(),
            avg_loss_val / count_val)
        data_iter_val.set_description(desc=desc_str_val, refresh=True)

        if loss_val.item() < minVal:
            minVal = loss_val.item()
            torch.save(net.state_dict(), '{}/trained_models/model.tar'.format(model_path))
            


