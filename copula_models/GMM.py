#for evaluation purpose, serve as baseline
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm, trange
from scipy import stats as st

import pytorch_lightning as pl
from torch_geometric.loader import DataLoader

class GMM:
    def __init__(self):
        """
        GMM confidence interval calculation.
        """
        self.results_dict = {}
    
    def predict(self, pred_trajs, agent_mask, epsilon):
        self.agent_mask = agent_mask
        self.results_dict[epsilon] = {"y_pred": pred_trajs}
        self.epsilon = epsilon

        return pred_trajs
    
    def calc_area(self, pred_trajs, cov_best_mode):
        '''average size over time steps and modes'''
        z_critical = st.norm.ppf(1 - self.epsilon / 2)
        loc, scale = pred_trajs.chunk(2, dim=-1)
        scale = scale.clone()
        with torch.no_grad():
            scale.clamp_(min=1e-6)

        area = torch.pi * scale[...,0] * scale[...,1] * z_critical**2 #[K, N, T]
        area_best_mode = area[cov_best_mode, torch.arange(area.shape[1])] #[N, T]

        return area_best_mode[self.agent_mask].mean() #average area for each timestep

    def calc_coverage_rate(self, test_pred_trajs, test_gt_trajs):
        '''check the ratio of covered predictions among all time steps and return the best mode'''
        z_critical = st.norm.ppf(1 - self.epsilon / 2)

        loc, scale = test_pred_trajs.chunk(2, dim=-1)
        scale = scale.clone()
        with torch.no_grad():
            scale.clamp_(min=1e-6)

        coverage = torch.abs(loc - test_gt_trajs.unsqueeze(0)) < z_critical*scale
        cov_alldim = coverage[...,0] & coverage[...,1] #[K, N, 50]
        cov_allts = torch.sum(cov_alldim, dim=-1)/cov_alldim.shape[-1] #[K, N]
        cov_best_mode = cov_allts.argmax(dim=0) #[N]
        cov_mode = cov_allts[cov_best_mode, torch.arange(cov_allts.shape[1])] #[N]

        return cov_mode[self.agent_mask].mean(), cov_best_mode

    def calc_coverage_all(self, test_pred_trajs, test_gt_trajs):
        '''check if prediction over all time steps are covered'''
        z_critical = st.norm.ppf(1 - self.epsilon / 2)

        loc, scale = test_pred_trajs.chunk(2, dim=-1)
        scale = scale.clone()
        with torch.no_grad():
            scale.clamp_(min=1e-6)

        coverage = torch.abs(loc - test_gt_trajs.unsqueeze(0)) < z_critical*scale
        cov_alldim = coverage[...,0] & coverage[...,1] #[K, N, 50]
        cov_allts = torch.all(cov_alldim, dim=-1) #[K, N]
        cov_mode = torch.any(cov_allts, dim=0) #[N], any mode achieves full coverage

        return torch.sum(cov_mode[self.agent_mask])/self.agent_mask.sum()

