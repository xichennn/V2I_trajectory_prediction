from argparse import ArgumentParser
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader

from datasets import V2XDataset
from models.v2x_model import V2X

# for copula
import numpy as np
import pandas as pd
import torch
from copulae.core import pseudo_obs
from tqdm import trange

#import model
from copula_models.GMM import GMM
from copula_models.vanillaCP import vanillaCP
from copula_models.copulaCPTS import copulaCPTS
from copula_models.utils import fix_randomness, prepare_data
import pickle

def experiment_GMM(gmm_data, eps):
    gmm = GMM()
    test_pred_trajs = gmm.predict(gmm_data.test_pred_trajs, gmm_data.test_agent_mask, epsilon=eps)
    
    coverage_rate, best_mode_rate = gmm.calc_coverage_rate(test_pred_trajs, gmm_data.test_gt_trajs)
    area_best_rate = gmm.calc_area(test_pred_trajs, best_mode_rate)
    coverage_all = gmm.calc_coverage_all(test_pred_trajs, gmm_data.test_gt_trajs)
    # save results
    gmm.results_dict[eps]['area_best_rate'] = area_best_rate
    gmm.results_dict[eps]['coverage_rate'] = coverage_rate
    gmm.results_dict[eps]['coverage_all'] = coverage_all
    gmm.results_dict[eps]['best_mode'] = best_mode_rate
    
    with open("uq_{}.pkl".format("gmm"), "wb") as f:
        pickle.dump(gmm.results_dict, f)
    # with open('uq_gmm.pkl', 'rb') as f:
    #     data = pickle.load(f)
    return area_best_rate, coverage_rate, coverage_all

def experiment_vanillaCP(vcp_data, eps, score_fun="stderr"):
    vanillacp = vanillaCP(vcp_data.cali_agent_mask, vcp_data.test_agent_mask)
    vanillacp.calibrate(vcp_data.cali_pred_trajs, vcp_data.cali_gt_trajs, score_fun) #"stderr" or "l2"

    test_pred_trajs, radius = vanillacp.predict(vcp_data.test_pred_trajs, epsilon=eps)

    if score_fun == "stderr":
        coverage_rate, best_mode_rate = vanillacp.calc_coverage_rate_stderr(test_pred_trajs, radius, vcp_data.test_gt_trajs)
        area_best_rate = vanillacp.calc_area_stderr(radius, best_mode_rate)
        coverage_all = vanillacp.calc_coverage_all_stderr(test_pred_trajs, radius, vcp_data.test_gt_trajs)
    else:
        coverage_rate, best_mode_rate = vanillacp.calc_coverage_rate_l2(test_pred_trajs, radius, vcp_data.test_gt_trajs)
        area_best_rate = vanillacp.calc_area_l2(radius, best_mode_rate)
        coverage_all = vanillacp.calc_coverage_all_l2(test_pred_trajs, radius, vcp_data.test_gt_trajs)
         
    #save results
    vanillacp.results_dict[eps]['area_best_rate'] = area_best_rate
    vanillacp.results_dict[eps]['coverage_rate'] = coverage_rate
    vanillacp.results_dict[eps]['coverage_all'] = coverage_all
    vanillacp.results_dict[eps]['best_mode'] = best_mode_rate
    
    with open("uq_{}_{}_{}.pkl".format("vcp", score_fun, eps), "wb") as f:
        pickle.dump(vanillacp.results_dict, f)

    return area_best_rate, coverage_rate, coverage_all

def experiment_copulaCPTS(copula_data, eps, score_fun="stderr"):
    copula = copulaCPTS(copula_data.test_agent_mask, copula_data.cali_agent_mask, copula_data.copula_agent_mask)
    copula.calibrate(copula_data.cali_pred_trajs, copula_data.cali_gt_trajs, score_fun) #"stderr" or "l2"

    test_pred_trajs, radius = copula.predict(copula_data.copula_pred_trajs, copula_data.copula_gt_trajs, copula_data.test_pred_trajs, epsilon=eps)
    if score_fun == "stderr":
        coverage_rate, best_mode_rate = copula.calc_coverage_rate_stderr(test_pred_trajs, radius, copula_data.test_gt_trajs)
        area_best_rate = copula.calc_area_stderr(radius, best_mode_rate)
        coverage_all = copula.calc_coverage_all_stderr(test_pred_trajs, radius, copula_data.test_gt_trajs)
    else:
        coverage_rate, best_mode_rate = copula.calc_coverage_rate_l2(test_pred_trajs, radius, copula_data.test_gt_trajs)
        area_best_rate = copula.calc_area_l2(radius, best_mode_rate)
        coverage_all = copula.calc_coverage_all_l2(test_pred_trajs, radius, copula_data.test_gt_trajs)
         
    #save results
    copula.results_dict[eps]['area_best_rate'] = area_best_rate
    copula.results_dict[eps]['coverage_rate'] = coverage_rate
    copula.results_dict[eps]['coverage_all'] = coverage_all
    copula.results_dict[eps]['best_mode'] = best_mode_rate
    
    with open("uq_{}_{}_{}.pkl".format("copula", score_fun, eps), "wb") as f:
        pickle.dump(copula.results_dict, f)

    return area_best_rate, coverage_rate, coverage_all


def experiment(model, valid_data, test_data):
    
    torch._C._set_mkldnn_enabled(False)  # this is to avoid a bug in pytorch that causes the code to crash

    UQ = {}

    copula = copulaCPTS(model, valid_data)
    copula.calibrate()
    UQ["copula"] = copula

    areas = {}
    coverages = {}

    epsilon_ls = np.linspace(0.05, 0.50, 10)

    for k, uqmethod in UQ.items():
        print(k)
        area = []
        coverage = []
        for eps in epsilon_ls:
            pred, box = uqmethod.predict(test_data, epsilon=eps)
            area.append(uqmethod.calc_area(box))
            pred = torch.tensor(pred)
            coverage.append(uqmethod.calc_coverage(box, pred, test_data))
        areas[k] = area
        coverages[k] = coverage

    with open("./trained/uq_%s.pkl" % name, "wb") as f:
        pickle.dump(UQ, f)
    with open("./trained/results_%s.pkl" % name, "wb") as f:
        pickle.dump((areas, coverages), f)

    return areas, coverages, (models, UQ)


def main():
    root = '/groups/klhead/xic/cyverse_data/cooperative-vehicle-infrastructure/vehicle-trajectories'
    ckpt_path = '/home/u6/xic/v2x_projects/V2X_graph_plugin/lightning_logs/version_36/checkpoints/epoch=56-step=45600.ckpt'
    model = V2X.load_from_checkpoint(checkpoint_path=ckpt_path, parallel=False)
    # checkpoint = torch.load("path/to/lightning/checkpoint.ckpt")
    # model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    trainer = pl.Trainer()

    # prepare data
    val_dataset = V2XDataset(root=root, split='val', local_radius=50)
    # for GMM
    gmm_data = prepare_data(model, val_dataset, cp="gmm", random_seed=1, batch_size=32, num_workers=19)
    gmm_data.prepare()
    # for vanillaCP
    vcp_data = prepare_data(model, val_dataset, cp="vcp", random_seed=1, batch_size=32, num_workers=19)
    vcp_data.prepare()
    # for copula
    copula_data = prepare_data(model, val_dataset, cp="copula", random_seed=1, batch_size=32, num_workers=19)
    copula_data.prepare()

    for eps in [0.2, 0.1, 0.05]:
        res_gmm = experiment_GMM(gmm_data, eps)
        res_vcp_stderr = experiment_vanillaCP(vcp_data, eps, score_fun="stderr")
        res_copula_stderr = experiment_copulaCPTS(copula_data, eps, score_fun="stderr")
        # res_copula_l2 = experiment_copulaCPTS(copula_data, eps, score_fun="l2")
        # res_vcp_l2 = experiment_vanillaCP(vcp_data, eps, score_fun="l2")

        print("alpha: ", eps)
        print("GMM results: ", res_gmm)
        # print("vanillaCP results l2", res_vcp_l2)
        # print("copula results l2", res_copula_l2)
        print("vanillaCP results stderr", res_vcp_stderr)
        print("copula results stderr", res_copula_stderr)
        print("Done!")

    # for i in range(3):
    #     res = experiment(model, valid_data, test_data)
    #     print("run " + str(i) + "done")
    #     del res

if __name__ == "__main__":
    main()

