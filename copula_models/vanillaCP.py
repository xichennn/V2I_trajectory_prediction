'''CP for each mode and each time step'''
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from copulae.core import pseudo_obs
from tqdm import tqdm, trange

import pytorch_lightning as pl
from torch_geometric.loader import DataLoader

class vanillaCP:
    def __init__(self, cali_agent_mask, test_agent_mask):
        """
        Copula conformal prediction with two-step calibration.
        """
        self.cali_agent_mask = cali_agent_mask
        self.test_agent_mask = test_agent_mask
        self.nonconformity = None
        self.results_dict = {}

    def score_function_stderr(self, pred_trajs, gt_trajs):
        '''
        pred_trajs: [6, N, 50, 4]
        pi: [N, 6]
        gt_trajs: [N, 50, 2]
        '''
        loc, scale = pred_trajs.chunk(2, dim=-1)
        scale = scale.clone()
        with torch.no_grad():
            scale.clamp_(min=1e-6)
        stderr = torch.abs(gt_trajs.unsqueeze(0) - loc) / scale #[K, N, T, 2]
        stderr_min, stderr_index = torch.min(stderr, dim=0)
        return stderr_min #[N, T, 2]
    
    def score_function_l2(self, pred_trajs, gt_trajs):
        '''
        pred_trajs: [6, N, 50, 4]
        pi: [N, 6]
        gt_trajs: [N, 50, 2]
        '''
        loc, scale = pred_trajs.chunk(2, dim=-1)
        scale = scale.clone()
        with torch.no_grad():
            scale.clamp_(min=1e-6)
        l2_norm = torch.norm((loc - gt_trajs.unsqueeze(0)), p=2, dim=-1) #[K, N, T]
        l2_norm_min, l2_norm_index = torch.min(l2_norm, dim=0)
        return l2_norm_min #[N, T]

    def score_function_l1(self, pred_trajs, gt_trajs):
        '''
        pred_trajs: [6, N, 50, 4]
        pi: [N, 6]
        gt_trajs: [N, 50, 2]
        '''
        loc, scale = pred_trajs.chunk(2, dim=-1)
        scale = scale.clone()
        with torch.no_grad():
            scale.clamp_(min=1e-6)
        l1_norm = torch.abs(loc - gt_trajs.unsqueeze(0)) #[K, N, T, 2]
        l1_norm_min, l1_norm_index = torch.min(l1_norm, dim=0)
        return l1_norm_min #[N, T, 2]

    def calibrate(self, cali_pred_trajs, cali_gt_trajs, score_fun="stderr"):
        self.fun_name = score_fun
        if self.fun_name == "stderr": 
            self.score_fun = self.score_function_stderr
            nonconformity = self.score_function_stderr(cali_pred_trajs, cali_gt_trajs) #[N, T, 2] 
        elif self.fun_name == "l2":
            self.score_fun = self.score_function_l2
            nonconformity = self.score_function_l2(cali_pred_trajs, cali_gt_trajs) #[N, T] 
        elif self.fun_name == "l1":
            self.score_fun = self.score_function_l1
            nonconformity = self.score_function_l1(cali_pred_trajs, cali_gt_trajs) #[N, T, 2]

        self.nonconformity = nonconformity[self.cali_agent_mask] #[N, T]

    def predict(self, test_pred_trajs, epsilon=0.1):

        nonconformity = self.nonconformity 
        n_calibration = nonconformity.shape[0] #[N, T]
        new_quantile = min(
            (n_calibration + 1.0)
            * (1 - (epsilon / test_pred_trajs.shape[-2]))
            / n_calibration,
            1,
        )
        # new_quantile = min(
        #     (n_calibration + 1.0)
        #     * (1 - epsilon)
        #     / n_calibration,
        #     1,
        # )
        
        if self.fun_name=="l2": #l2 score
            radius = [
                torch.quantile(nonconformity[:, r], new_quantile)
                for r in range(nonconformity.shape[1])
            ] #[T]
            radius = torch.tensor(radius)
        elif self.fun_name=="stderr": #stderr score
            radius = [[
                torch.quantile(nonconformity[:, r, i], new_quantile)
                for r in range(nonconformity.shape[1])
            ] for i in range(nonconformity.shape[2])]
            radius = torch.tensor(radius).transpose(0, 1) #[T, 2]
            radius = torch.mul(test_pred_trajs[:,self.test_agent_mask,:,2:], radius.view(1, 1, *radius.shape)) #[K,N,T,2]
        elif self.fun_name=="l1": #l1 score
            radius = [[
                torch.quantile(nonconformity[:, r, i], new_quantile)
                for r in range(nonconformity.shape[1])
            ] for i in range(nonconformity.shape[2])]
            radius = torch.tensor(radius).transpose(0, 1) #[T, 2]
    
        self.results_dict[epsilon] = {"y_pred": test_pred_trajs, "radius": radius, "agent_mask":self.test_agent_mask}

        return test_pred_trajs, radius

    def calc_area_l2(self, radius, best_cov_mode):
        '''radius: [T]'''
        area = sum([np.pi * r**2 for r in radius])/len(radius)

        return area
    def calc_area_l1(self, radius, best_cov_mode):
        '''radius: [T, 2]'''
        area = torch.pi * radius[...,0] * radius[...,1]

        return area.mean()

    def calc_area_stderr(self, radius, cov_best_mode):
        area = torch.pi * radius[...,0] * radius[...,1] #[K, N, T]
        area_best_mode = area[cov_best_mode, torch.arange(area.shape[1])] #[N, T]

        return area_best_mode.mean() #average area for each timestep

    def calc_coverage_rate_l2(self, test_pred_trajs, radius, test_gt_trajs):
        '''radius: [K, T]'''
        test_pred_trajs = test_pred_trajs[:,self.test_agent_mask]
        test_gt_trajs = test_gt_trajs[self.test_agent_mask]
        testnonconformity = self.score_fun(test_pred_trajs, test_gt_trajs) #[N, T] 

        circle_covs = []
        for j in range(test_gt_trajs.shape[-2]):
            circle_covs.append(testnonconformity[:, j] < radius[j])
        circle_covs = torch.stack(circle_covs, dim=0) #[T, N]

        cov_best_mode = 0
        cov_mode = circle_covs.sum(dim=1)/circle_covs.shape[1] #[T]

        return cov_mode.mean(), cov_best_mode

    def calc_coverage_all_l2(self, test_pred_trajs, radius, test_gt_trajs):
        '''radius: [K, T]'''
        test_pred_trajs = test_pred_trajs[:,self.test_agent_mask]
        test_gt_trajs = test_gt_trajs[self.test_agent_mask]
        testnonconformity = self.score_fun(test_pred_trajs, test_gt_trajs) #[N, T] 

        circle_covs = []
        for j in range(test_gt_trajs.shape[-2]):
            circle_covs.append(testnonconformity[:, j] < radius[j])
        circle_covs = torch.stack(circle_covs, dim=0) #[T, N]

        cov_all = torch.all(circle_covs, dim=0) #[N]
        
        return torch.sum(cov_all)/cov_all.shape[0]
    
    def calc_coverage_rate_stderr(self, test_pred_trajs, radius, test_gt_trajs):
        '''check the ratio of covered predictions among all time steps and return the best mode'''
        test_pred_trajs = test_pred_trajs[:,self.test_agent_mask]
        test_gt_trajs = test_gt_trajs[self.test_agent_mask]
        loc, scale = test_pred_trajs.chunk(2, dim=-1)

        coverage = torch.abs(loc - test_gt_trajs.unsqueeze(0)) < radius #[K, N, T, 2]
        cov_alldim = coverage[...,0] & coverage[...,1] #[K, N, 50]
        cov_allts = torch.sum(cov_alldim, dim=-1)/cov_alldim.shape[-1] #[K, N]
        cov_best_mode = cov_allts.argmax(dim=0) #[N]
        cov_mode = cov_allts[cov_best_mode, torch.arange(cov_allts.shape[1])] #[N]

        return cov_mode.mean(), cov_best_mode

    def calc_coverage_all_stderr(self, test_pred_trajs, radius, test_gt_trajs):
        '''check if prediction over all time steps are covered'''
        test_pred_trajs = test_pred_trajs[:,self.test_agent_mask]
        test_gt_trajs = test_gt_trajs[self.test_agent_mask]
        loc, scale = test_pred_trajs.chunk(2, dim=-1)

        coverage = torch.abs(loc - test_gt_trajs.unsqueeze(0)) < radius
        cov_alldim = coverage[...,0] & coverage[...,1] #[K, N, 50]
        cov_allts = torch.all(cov_alldim, dim=-1) #[K, N]
        cov_mode = torch.any(cov_allts, dim=0) #[N], any mode achieves full coverage

        return torch.sum(cov_mode)/cov_mode.shape[0]


