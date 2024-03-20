# Copyright (c) 2022, Zikang Zhou. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from coop_models.modelBase import TemporalEncoder, AA_GAT, cross_graph_attention, ALEncoder, PredictionDecoder
from dataloader.temporal_data import TemporalData, get_lane_features
from coop_models.utils import MLP, DistanceDropEdge, V2XDropEdge

from torch_geometric.utils import subgraph
import torchvision
from torch.optim import Adam, AdamW
from utils import ScheduledOptim
from Coop_plugin.metrics.loss import NLLloss
import math


class Coop(pl.LightningModule):
    def __init__(self,
                 ip_dim: int=2,
                 historical_steps: int=50,
                 embed_dim: int=16,
                 num_heads: int=8,
                 num_temporal_layers: int=4,
                 out_dim: int=64,
                 dropout: float=0.1,
                 model_radius: float=50,
                 iou_thresh: float=0.3,
                 num_mode: int=5,
                 op_len: int=50,
                 lr: float=1e-3,
                 weight_decay: float=1e-4,
                 betas = (0.9, 0.999),
                 warmup_epoch: int=10,
                 lr_update_freq: int=10,
                 lr_decay_rate: float=0.9,
                 device = "cpu",
                 use_variance: bool = True,
                 **kwargs) -> None:
        super(Coop, self).__init__()
        self.save_hyperparameters()
        self.historical_steps = historical_steps
        self.model_radius = model_radius
        self.lr = lr
        self.weight_decay = weight_decay
        self.betas = betas
        self.warmup_epoch = warmup_epoch
        self.lr_update_freq = lr_update_freq
        self.lr_decay_rate = lr_decay_rate
        # self.ip_emb = MLP(ip_dim, embed_dim)
        self.v2x_graph = AA_GAT(node_dim = ip_dim, 
                                  embed_dim = embed_dim, 
                                  out_dim = out_dim, 
                                  edge_attr_dim = ip_dim, 
                                  device = device, 
                                  num_heads=num_heads, 
                                  dropout=dropout)
        self.car_graph = AA_GAT(node_dim = ip_dim, 
                                  embed_dim = embed_dim, 
                                  out_dim = out_dim, 
                                  edge_attr_dim = ip_dim, 
                                  device = device, 
                                  num_heads=num_heads, 
                                  dropout=dropout)
        self.cross_graph_attention = cross_graph_attention(embed_dim=embed_dim*num_heads,
                                                           num_heads=num_heads,
                                                           dropout=dropout)

        self.temp_encoder = TemporalEncoder(historical_steps=historical_steps,
                                    embed_dim=out_dim,
                                    num_heads=num_heads,
                                    num_layers=num_temporal_layers,
                                    dropout=dropout)
        # self.dm = DAIRV2XMap()
        self.al_encoder = ALEncoder(node_dim=ip_dim,
                                    edge_dim=ip_dim,
                                    embed_dim=out_dim,
                                    num_heads=num_heads,
                                    dropout=dropout)
        self.drop_edge_distance = DistanceDropEdge(model_radius)
        self.drop_edge_vic = V2XDropEdge(iou_thresh)
        self.decoder = PredictionDecoder(encoding_size=out_dim,
                                    hidden_size=out_dim,
                                    num_modes=num_mode,
                                    op_len=op_len,
                                    use_variance=use_variance)
        self.criterion = NLLloss(alpha=0.5, use_variance=False, device=device)

    def forward(self, v2x_data: TemporalData):
        #fuse information from two graphs according to iou_matrix at each timestamp
        v2x_graph_out = [None] * self.historical_steps

        for t in range(self.historical_steps):
            valid_t = ~v2x_data["padding_mask"][:,t] #nonzero coordinates
            #get subgraph accroding to padding_mask and iou match

            v2x_data[f'edge_index_{t}'], _ = subgraph(subset=valid_t, edge_index=v2x_data.edge_index)
            v2x_data[f'edge_attr_{t}'] = \
                v2x_data['positions'][v2x_data[f'edge_index_{t}'][0], t] - v2x_data['positions'][v2x_data[f'edge_index_{t}'][1], t]

            # get vic graph encoding
            v2x_edge_index_t, v2x_edge_attr_t = self.drop_edge_distance(v2x_data[f'edge_index_{t}'], v2x_data[f'edge_attr_{t}'])
            (merged_edge_index_t, merged_edge_attr_t, matched_car_infra_nodes) = \
            self.drop_edge_vic(v2x_edge_index_t, v2x_edge_attr_t, v2x_data["positions"], v2x_data["width"], v2x_data["height"], v2x_data["source"], t)

            v2x_graph_out[t] = self.v2x_graph(X=v2x_data.x[:, t], edge_index=merged_edge_index_t, edge_attr=merged_edge_attr_t, matched_car_infra_nodes=matched_car_infra_nodes) #[N1, embed_size]
                                        # bos_mask=infra_data['bos_mask'][:, t], rotate_mat=rotate_imat)
        

            #use mask for overlapping agents
        v2x_graph_out = torch.stack(v2x_graph_out)  # [T, N, D]
        # #temporal encoding
        temp_out = self.temp_encoder(v2x_graph_out, padding_mask=v2x_data['padding_mask'][:, : self.historical_steps]) #[N, D]

        #AL encoding
        al_edge_index, al_edge_attr = self.drop_edge_distance(v2x_data.lane_actor_index, v2x_data.lane_actor_vectors)
        al_out = self.al_encoder(x=(v2x_data.lane_vectors, temp_out), edge_index=al_edge_index, edge_attr=al_edge_attr,
                              is_intersections=v2x_data.is_intersections, turn_directions=v2x_data.turn_directions,
                              traffic_controls=v2x_data.traffic_controls, rotate_mat=v2x_data.rotate_imat.float())

        #decoder
        out = self.decoder(al_out) #traj, log_probs

        return out 
    
    def get_transformed_vectorized_coords(self, infra_data: TemporalData, car_data: TemporalData):
        # transform the global coords into AV centered coords
        rotate_mat = torch.tensor([[torch.cos(car_data.theta), -torch.sin(car_data.theta)],
                                [torch.sin(car_data.theta), torch.cos(car_data.theta)]], dtype=torch.float64)
        infra_data["positions_local"] = torch.matmul(infra_data.positions - car_data.origin.double(), rotate_mat)
        car_data["positions_local"] = torch.matmul(car_data.positions - car_data.origin.double(), rotate_mat)
        
        #vectorize trajs 
        infra_data["x"][:, 1: 50] = torch.where((infra_data["padding_mask"][:, : 49] | infra_data["padding_mask"][:, 1: 50]).unsqueeze(-1),
                              torch.zeros(infra_data.num_nodes, 49, 2, dtype = infra_data["positions_local"].dtype),
                              infra_data["positions_local"][:, 1: 50] - infra_data["positions_local"][:, : 49]) # difference
        infra_data["x"][:, 0] = torch.zeros(infra_data.num_nodes, 2)
        infra_data["y"] = infra_data["positions_local"][:, 50:] - infra_data["positions_local"][:, 49].unsqueeze(-2)
        car_data["x"][:, 1: 50] = torch.where((car_data["padding_mask"][:, : 49] | car_data["padding_mask"][:, 1: 50]).unsqueeze(-1),
                              torch.zeros(car_data.num_nodes, 49, 2, dtype = car_data["positions_local"].dtype),
                              car_data["positions_local"][:, 1: 50] - car_data["positions_local"][:, : 49]) # difference
        car_data["x"][:, 0] = torch.zeros(car_data.num_nodes, 2)
        car_data["y"] = car_data["positions_local"][:, 50:] - car_data["positions_local"][:, 49].unsqueeze(-2)

        #rotate all trajs such that they are facing the positive x-axis
        rotate_imat = torch.empty(car_data.num_nodes+infra_data.num_nodes, 2, 2, device=self.device)
        sin_vals = torch.sin(torch.cat((car_data['rotate_angles'], infra_data['rotate_angles'])))
        cos_vals = torch.cos(torch.cat((car_data['rotate_angles'], infra_data['rotate_angles'])))
        rotate_imat[:, 0, 0] = cos_vals
        rotate_imat[:, 0, 1] = -sin_vals
        rotate_imat[:, 1, 0] = sin_vals
        rotate_imat[:, 1, 1] = cos_vals
        car_data.x = torch.bmm(car_data.x, rotate_imat[:car_data.num_nodes]) 
        infra_data.x = torch.bmm(infra_data.x, rotate_imat[car_data.num_nodes:])
        return infra_data, car_data, rotate_mat, rotate_imat
    
    def get_iou_mat(self, infra_data, car_data, t, valid_cars_t, valid_infra_t, iou_thresh=0.3):
        # get the graph match matrix at timestep t via iou
        iou_matrix_batch = [None] * len(car_data.seq_id)
        for batch in range(len(car_data.seq_id)):
            valid_infra_t_batch = valid_infra_t[infra_data.batch == batch]
            infra_bbox_t_batch = torch.stack((infra_data["positions"][infra_data.batch == batch][valid_infra_t_batch,t,0] - infra_data["width"][infra_data.batch == batch][valid_infra_t_batch,t], 
                                infra_data["positions"][infra_data.batch == batch][valid_infra_t_batch,t,1] - infra_data["height"][infra_data.batch == batch][valid_infra_t_batch,t],
                                infra_data["positions"][infra_data.batch == batch][valid_infra_t_batch,t,0] + infra_data["width"][infra_data.batch == batch][valid_infra_t_batch,t], 
                                infra_data["positions"][infra_data.batch == batch][valid_infra_t_batch,t,1] + infra_data["height"][infra_data.batch == batch][valid_infra_t_batch,t]), dim=1)
            valid_cars_t_batch = valid_cars_t[car_data.batch == batch]
            car_bbox_t_batch = torch.stack((car_data["positions"][car_data.batch == batch][valid_cars_t_batch,t,0] - car_data["width"][car_data.batch == batch][valid_cars_t_batch,t], 
                                car_data["positions"][car_data.batch == batch][valid_cars_t_batch,t,1] - car_data["height"][car_data.batch == batch][valid_cars_t_batch,t],
                                car_data["positions"][car_data.batch == batch][valid_cars_t_batch,t,0] + car_data["width"][car_data.batch == batch][valid_cars_t_batch,t], 
                                car_data["positions"][car_data.batch == batch][valid_cars_t_batch,t,1] + car_data["height"][car_data.batch == batch][valid_cars_t_batch,t]), dim=1)
            iou_t = torchvision.ops.box_iou(infra_bbox_t_batch, car_bbox_t_batch)
            iou_matrix = iou_t > iou_thresh
            iou_matrix_batch[batch] = iou_matrix


        infra_bbox_t = torch.stack((infra_data["positions"][valid_infra_t,t,0] - infra_data["width"][valid_infra_t,t], 
                                infra_data["positions"][valid_infra_t,t,1] - infra_data["height"][valid_infra_t,t],
                                infra_data["positions"][valid_infra_t,t,0] + infra_data["width"][valid_infra_t,t], 
                                infra_data["positions"][valid_infra_t,t,1] + infra_data["height"][valid_infra_t,t]), dim=1)
        car_bbox_t = torch.stack((car_data["positions"][valid_cars_t,t,0] - car_data["width"][valid_cars_t,t], 
                                car_data["positions"][valid_cars_t,t,1] - car_data["height"][valid_cars_t,t],
                                car_data["positions"][valid_cars_t,t,0] + car_data["width"][valid_cars_t,t], 
                                car_data["positions"][valid_cars_t,t,1] + car_data["height"][valid_cars_t,t]), dim=1)
        iou_t = torchvision.ops.box_iou(infra_bbox_t, car_bbox_t)
        iou_matrix = iou_t > iou_thresh

        return iou_matrix

    def training_step(self, data, batch_idx):
        pred = self(data)
        loss = self.criterion(pred['traj'][:,:,:,:2], data.y, pred['log_probs'])
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=1)
        return loss

    def validation_step(self, data, batch_idx):
        pred_val = self(data)
        loss_val = self.criterion(pred_val['traj'][:,:,:,:2],
                                 data.y, pred_val['log_probs'])
        self.log('val_loss', loss_val, prog_bar=True, on_step=False, on_epoch=True, batch_size=1)

        # y_hat_agent = y_hat[:, data['agent_index'], :, : 2]
        # y_agent = data.y[data['agent_index']]
        # fde_agent = torch.norm(y_hat_agent[:, :, -1] - y_agent[:, -1], p=2, dim=-1)
        # best_mode_agent = fde_agent.argmin(dim=0)
        # y_hat_best_agent = y_hat_agent[best_mode_agent, torch.arange(data.num_graphs)]
        # self.minADE.update(y_hat_best_agent, y_agent)
        # self.minFDE.update(y_hat_best_agent, y_agent)
        # self.minMR.update(y_hat_best_agent, y_agent)
        # self.log('val_minADE', self.minADE, prog_bar=True, on_step=False, on_epoch=True, batch_size=y_agent.size(0))
        # self.log('val_minFDE', self.minFDE, prog_bar=True, on_step=False, on_epoch=True, batch_size=y_agent.size(0))
        # self.log('val_minMR', self.minMR, prog_bar=True, on_step=False, on_epoch=True, batch_size=y_agent.size(0))

    def configure_optimizers(self):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.MultiheadAttention, nn.LSTM, nn.GRU)
        blacklist_weight_modules = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.Embedding)
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                full_param_name = '%s.%s' % (module_name, param_name) if module_name else param_name
                if 'bias' in param_name:
                    no_decay.add(full_param_name)
                elif 'weight' in param_name:
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    elif isinstance(module, blacklist_weight_modules):
                        no_decay.add(full_param_name)
                elif not ('weight' in param_name or 'bias' in param_name):
                    no_decay.add(full_param_name)
        param_dict = {param_name: param for param_name, param in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0
        assert len(param_dict.keys() - union_params) == 0

        optim_groups = [
            {"params": [param_dict[param_name] for param_name in sorted(list(decay))],
             "weight_decay": self.weight_decay},
            {"params": [param_dict[param_name] for param_name in sorted(list(no_decay))],
             "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(optim_groups, lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.T_max, eta_min=0.0)

        # optimizer = AdamW(net.parameters(), lr=self.lr, betas=self.betas, weight_decay=self.weight_decay)
        # scheduler = ScheduledOptim(
        #     optimizer,
        #     self.lr,
        #     n_warmup_epoch=self.warmup_epoch,
        #     update_rate=self.lr_update_freq,
        #     decay_rate=self.lr_decay_rate
        # )
        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('Coop')
        parser.add_argument('--ip_dim', type=int, default=2)
        parser.add_argument('--historical_steps', type=int, default=50)
        parser.add_argument('--num_heads', type=int, default=8)
        parser.add_argument('--embed_dim', type=int, default=16)
        parser.add_argument('--num_temporal_layers', type=int, default=4)
        parser.add_argument('--out_dim', type=int, default=64)
        parser.add_argument('--dropout', type=float, default=0.1)
        parser.add_argument('--model_radius', type=float, default=50)
        parser.add_argument('--iou_thresh', type=float, default=0.3)
        parser.add_argument('--num_modes', type=int, default=6)
        parser.add_argument('--op_len', type=int, default=50)
        parser.add_argument('--lr', type=float, default=3e-4)
        parser.add_argument('--weight_decay', type=float, default=1e-4)
        parser.add_argument('--T_max', type=int, default=64)
        parser.add_argument('--use_variance', type=bool, default=True)
        return parent_parser
    