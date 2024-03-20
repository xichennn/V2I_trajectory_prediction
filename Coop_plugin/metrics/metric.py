#source: https://github.com/nachiket92/PGP/blob/main/metrics/utils.py
import torch
from typing import Tuple

def ade(traj: torch.Tensor, traj_gt: torch.Tensor):
    ls = torch.norm(traj - traj_gt, p=2, dim=-1).mean(dim=-1).mean()

    return ls

def fde(traj: torch.Tensor, traj_gt: torch.Tensor):
    ls = torch.norm(traj[:, -1] - traj_gt[:, -1], p=2, dim=-1).mean()

    return ls

def mr(traj: torch.Tensor, traj_gt: torch.Tensor, miss_threshold: torch.Tensor):
    ls = (torch.norm(traj[:, -1] - traj_gt[:, -1], p=2, dim=-1) > miss_threshold).sum()

    return ls/traj.shape[0]


def min_ade(traj: torch.Tensor, traj_gt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes average displacement error for the best trajectory in a set, with respect to ground truth
    :param traj: predictions, shape [num_vehs, num_modes, op_len, 2]
    :param traj_gt: ground truth trajectory, shape [num_vehs, op_len, 2]
    :return errs, inds: errors and indices for modes with min error, shape [num_vehs]
    """
    num_modes = traj.shape[1]
    op_len = traj.shape[2]

    traj_gt_rpt = traj_gt.unsqueeze(1).repeat(1, num_modes, 1, 1)
    # masks_rpt = masks.unsqueeze(1).repeat(1, num_modes, 1)

    err = (traj_gt_rpt - traj[:, :, :, 0:2])
    err = torch.pow(err, exponent=2)
    err = torch.sum(err, dim=3)
    err = torch.pow(err, exponent=0.5)
    err = torch.sum(err, dim=2) / op_len
    
    # err[stat_idx,:] = err[stat_idx,:]*10000

    err, inds = torch.min(err, dim=1)

    return err, inds

def traj_nll(pred_dist: torch.Tensor, traj_gt: torch.Tensor):
    """
    Computes negative log likelihood of ground truth trajectory under a predictive distribution with a single mode,
    with a bivariate Gaussian distribution predicted at each time in the prediction horizon

    :param pred_dist: parameters of a bivariate Gaussian distribution, shape [num_vehs, op_len, 5]
    :param traj_gt: ground truth trajectory, shape [num_vehs, op_len, 2]
    :return:
    """
    # op_len = pred_dist.shape[1]
    # mu_x = pred_dist[:, :, 0]
    # mu_y = pred_dist[:, :, 1]
    # x = traj_gt[:, :, 0]
    # y = traj_gt[:, :, 1]

    # sig_x = pred_dist[:, :, 2]
    # sig_y = pred_dist[:, :, 3]
    # rho = pred_dist[:, :, 4]
    # ohr = torch.pow(1 - torch.pow(rho, 2), -0.5)

    # nll = 0.5 * torch.pow(ohr, 2) * \
    #     (torch.pow(sig_x, 2) * torch.pow(x - mu_x, 2) +
    #      torch.pow(sig_y, 2) * torch.pow(y - mu_y, 2) -
    #      2 * rho * torch.pow(sig_x, 1) * torch.pow(sig_y, 1) * (x - mu_x) * (y - mu_y))\
    #     - torch.log(sig_x * sig_y * ohr) + 1.8379

    # nll[nll.isnan()] = 0
    # nll[nll.isinf()] = 0

    # nll = torch.sum(nll, dim=1) / op_len
    pred_loc = pred_dist[:,:,:2]
    pred_var = pred_dist[:,:,2:4]

    nll = torch.sum(0.5 * torch.log(pred_var) + 0.5 * torch.div(torch.square(traj_gt - pred_loc), pred_var) +\
                     0.5 * torch.log(2 * torch.tensor(3.14159265358979323846)))


    return nll

def NLLloss(y_pred, y_true, log_probs, anchors):
    """
    params:
    :y_pred: [N_T, N_M, N_B, 2]  
    :y_true: [N_T, N_B, 2]
    :log_probs: probability for each mode [N_B, N_M]
    :anchors: [N_M, N_T,2]
    where N_B is batch_size, N_M is num_modes, N_T is target_len 
    """

    
    batch_size = y_true.shape[1]
    trajectories = y_pred
    anchor_probs = log_probs
    
    #find the nearest anchor mode to y_true
    #[1, N_M, N_T,2] - [N_B, N_M, N_T, 2] = [N_B, N_M, N_T, 2]
    distance_to_anchors = torch.sum(torch.linalg.vector_norm(anchors.unsqueeze(0) - y_true.permute(1,0,2).unsqueeze(1),
                dim=(-1)),dim=-1) #[N_B, N_M]

    nearest_mode = distance_to_anchors.argmin(dim=-1) #[N_B]
    nearest_mode_indices = torch.stack([torch.arange(batch_size,dtype=torch.int64),nearest_mode],dim=-1)

    loss_cls = -log_probs[torch.arange(batch_size),nearest_mode].squeeze() #[N_B]
    
    #trajectories_xy: [N_B, N_M, N_T, 2]   
    #nearest_trajs: [N_B, N_T, 2]
    #residual_trajs: [N_B, N_T, 2]
    trajectories_xy = y_pred.permute(2,1,0,3)[...,:2] + anchors.unsqueeze(0)
    nearest_trajs = trajectories_xy[torch.arange(batch_size),nearest_mode,:,:].squeeze()
    residual_trajs = y_true.permute(1,0,2) - nearest_trajs

    loss_reg = torch.mean(torch.square(residual_trajs[:,:,0])+torch.square(residual_trajs[:,:,1]), dim=-1)
    dx = residual_trajs[:,:,0]
    dy = residual_trajs[:,:,1]
    
    total_loss = torch.mean(loss_cls+loss_reg)

    return loss_cls, loss_reg