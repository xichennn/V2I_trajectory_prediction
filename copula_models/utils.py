import random
import math
import warnings

import numpy as np
import torch

import pytorch_lightning as pl
from torch_geometric.loader import DataLoader

__all__ = ["fix_randomness", "DimensionError", "get_device"]


def fix_randomness(seed=0):
    """
    Fix the random seed for python, torch, numpy.

    :param seed: the random seed
    """
    np.random.seed(seed=seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)

def calculate_conformal_value(scores, alpha, default_q_hat = torch.inf):
    """
    Calculate the 1-alpha quantile of scores.
    
    :param scores: non-conformity scores.
    :param alpha: a significance level.
    
    :return: the threshold which is use to construct prediction sets.
    """
    if default_q_hat == "max":
        default_q_hat = torch.max(scores)
    if alpha >= 1 or alpha <= 0:
            raise ValueError("Significance level 'alpha' must be in [0,1].")
    if len(scores) == 0:
        warnings.warn(
            f"The number of scores is 0, which is a invalid scores. To avoid program crash, the threshold is set as {default_q_hat}.")
        return default_q_hat
    N = scores.shape[0]
    qunatile_value = math.ceil(N + 1) * (1 - alpha) / N
    if qunatile_value > 1:
        warnings.warn(
            f"The value of quantile exceeds 1. It should be a value in [0,1]. To avoid program crash, the threshold is set as {default_q_hat}.")
        return default_q_hat

    return torch.quantile(scores, qunatile_value, dim=0).to(scores.device)

def get_trajs(model, loader):
    trainer = pl.Trainer()
    with torch.no_grad():
        predicts_list = trainer.predict(model, loader)
    pred_trajs, pi = [], []
    for i in range(len(predicts_list)):
        pred_traj, p =  predicts_list[i]
        pred_trajs.append(pred_traj) #[K, N, 50, 4]
        pi.append(p)
    pred_trajs = torch.cat(pred_trajs, 1) #[K, N, 50, 4]
    pi = torch.cat(pi, 0) #[N, K]

    gt_trajs = []
    for i in range(len(loader.dataset)):
        rotate_mat = torch.empty(loader.dataset[i].num_nodes, 2, 2)
        sin_vals = torch.sin(loader.dataset[i]['rotate_angles'])
        cos_vals = torch.cos(loader.dataset[i]['rotate_angles'])
        rotate_mat[:, 0, 0] = cos_vals
        rotate_mat[:, 0, 1] = -sin_vals
        rotate_mat[:, 1, 0] = sin_vals
        rotate_mat[:, 1, 1] = cos_vals
        gt_trajs.append(torch.bmm(loader.dataset[i].y, rotate_mat)) #[N, 50, 2]
    gt_trajs = torch.cat(gt_trajs, 0) #[N, 50, 2]

    return pred_trajs, pi, gt_trajs

def get_agent_index(loader):
    agent_index = []
    node_count=0
    for i in range(len(loader.dataset)):
        agent_index.append(loader.dataset[i].agent_index+node_count) #[N']
        node_count += loader.dataset[i].y.shape[0]
    # agent_index = torch.cat(agent_index, 0) #[N, 50, 2]
    agent_mask =torch.zeros(node_count, dtype=bool)
    agent_mask[agent_index] = True
    return agent_mask

class prepare_data:
    def __init__(self, model, val_dataset, cp="copula", random_seed=1, batch_size=32, num_workers=1):
        self.model = model
        self.val_dataset = val_dataset
        self.cp = cp
        self.random_seed = random_seed
        self.batch_size = batch_size
        self.num_workers = num_workers
    
    def prepare(self):
        fix_randomness(self.random_seed)
        #split the data
        idx = torch.arange(len(self.val_dataset))

        # Split the shuffled indices
        split = int(0.2 * len(idx))

        test_data = self.val_dataset[idx[:split]]
        valid_data = self.val_dataset[idx[split:]]
        
        test_loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=self.num_workers)
        self.test_pred_trajs, self.test_pi, self.test_gt_trajs = get_trajs(self.model, test_loader)
        self.test_agent_mask = get_agent_index(test_loader)

        if self.cp=="copula":
            split_cp = 0.5
            size = len(valid_data)
            halfsize = int(split_cp * size)

            idx = np.random.choice(range(size), halfsize, replace=False)

            cali_loader = DataLoader(valid_data[idx], batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=self.num_workers)
            copula_loader = DataLoader(valid_data[list(set(range(size)) - set(idx))], batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=self.num_workers)
            
            self.cali_agent_mask = get_agent_index(cali_loader)
            self.copula_agent_mask = get_agent_index(copula_loader)

            self.cali_pred_trajs, self.cali_pi, self.cali_gt_trajs = get_trajs(self.model, cali_loader)
            self.copula_pred_trajs, self.copula_pi, self.copula_gt_trajs = get_trajs(self.model, copula_loader)

        else:
            cali_loader = DataLoader(valid_data, batch_size=self.batch_size, shuffle=False, pin_memory=True)
            self.cali_agent_mask = get_agent_index(cali_loader)
            self.cali_pred_trajs, self.cali_pi, self.cali_gt_trajs = get_trajs(self.model, cali_loader)






