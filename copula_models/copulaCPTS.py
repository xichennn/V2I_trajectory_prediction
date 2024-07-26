import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from copulae.core import pseudo_obs
from tqdm import tqdm, trange

import pytorch_lightning as pl
from torch_geometric.loader import DataLoader

class CP(nn.Module):
    def __init__(self, dimension, epsilon):
        super(CP, self).__init__()
        self.alphas = nn.Parameter(torch.ones(dimension))
        self.epsilon = epsilon
        self.relu = torch.nn.ReLU()

    def forward(self, pseudo_data):
        coverage = torch.mean(
            torch.relu(
                torch.prod(torch.sigmoid((self.alphas - pseudo_data) * 1000), dim=1)
            )
        )
        return torch.abs(coverage - 1 + self.epsilon)

def search_alpha(alpha_input, epsilon, epochs=500):
    # pseudo_data = torch.tensor(pseudo_obs(alpha_input))
    pseudo_data = alpha_input.clone().detach()
    dim = alpha_input.shape[-1]
    cp = CP(dim, epsilon)
    optimizer = torch.optim.Adam(cp.parameters(), weight_decay=1e-4)

    with trange(epochs, desc="training", unit="epochs") as pbar:
        for i in pbar:
            optimizer.zero_grad()
            loss = cp(pseudo_data)

            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=loss.detach().numpy())

    return cp.alphas

def empirical_copula_loss(x, data, epsilon):
    pseudo_data = pseudo_obs(data)
    return np.fabs(
        np.mean(
            np.all(
                np.less_equal(pseudo_data, np.array([x] * pseudo_data.shape[1])), axis=1
            )
        )
        - 1
        + epsilon
    )

class copulaCPTS:
    def __init__(self, test_agent_mask, cali_agent_mask, copula_agent_mask):
        """
        Copula conformal prediction with two-step calibration.
        """
        self.test_agent_mask = test_agent_mask
        self.cali_agent_mask = cali_agent_mask
        self.copula_agent_mask = copula_agent_mask
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
        return stderr_min

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
        return l2_norm_min

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

    def get_radius(self, scores, nonconformity, epsilon):
        '''scores: [N1, T], nonconformity: [N2, T]'''
        alphas = []
        for i in range(scores.shape[0]):
            a_ = scores[i] > nonconformity
            a = torch.sum(a_, dim=0)/a_.shape[0] #[T,]
            alphas.append(a)
        alphas = torch.stack(alphas, dim=0) #[N1, T]

        threshold = search_alpha(alphas, epsilon, epochs=800) #[T,]

        mapping = {
            i: sorted(nonconformity[:, i].tolist()) for i in range(alphas.shape[1])
        }

        quantile = []
        mapping_shape = nonconformity.shape[0]

        for i in range(alphas.shape[1]):
            idx = int(threshold[i] * mapping_shape) + 1
            if idx >= mapping_shape:
                idx = mapping_shape - 1
            quantile.append(mapping[i][idx])
        radius = torch.Tensor(quantile)

        return radius

    def predict(self, copula_pred_trajs, copula_gt_trajs, test_pred_trajs, epsilon=0.1):
        copulanonconformity = self.score_fun(copula_pred_trajs, copula_gt_trajs)[self.copula_agent_mask] #[N, T, 2]

        if self.fun_name=="l2": #l2 score
            radius = self.get_radius(copulanonconformity, self.nonconformity, epsilon)
        elif self.fun_name=="stderr":
            radius = [self.get_radius(copulanonconformity[...,i], self.nonconformity[...,i], epsilon/2) for i in range(2)]
            radius = torch.stack(radius, dim=0).transpose(0,1) #[50,2]
            radius = torch.mul(test_pred_trajs[:,self.test_agent_mask,:,2:], radius.view(1, 1, *radius.shape)) #[K,N,T,2]
        elif self.fun_name=="l1":
            radius = [self.get_radius(copulanonconformity[...,i], self.nonconformity[...,i], epsilon/2) for i in range(2)]
            radius = torch.stack(radius, dim=0).transpose(0,1) #[50,2]

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
        '''radius: [T]'''
        test_pred_trajs = test_pred_trajs[:,self.test_agent_mask]
        test_gt_trajs = test_gt_trajs[self.test_agent_mask]
        testnonconformity = self.score_fun(test_pred_trajs, test_gt_trajs) #[K, N, T] 

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


