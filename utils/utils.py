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
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch_geometric.data import Data


class TemporalData(Data):

    def __init__(self,
                 x: Optional[torch.Tensor] = None,
                 positions: Optional[torch.Tensor] = None,
                 edge_index: Optional[torch.Tensor] = None,
                 edge_attrs: Optional[List[torch.Tensor]] = None,
                 y: Optional[torch.Tensor] = None,
                 num_nodes: Optional[int] = None,
                 padding_mask: Optional[torch.Tensor] = None,
                 bos_mask: Optional[torch.Tensor] = None,
                 rotate_angles: Optional[torch.Tensor] = None,
                 lane_vectors: Optional[torch.Tensor] = None,
                 is_intersections: Optional[torch.Tensor] = None,
                 turn_directions: Optional[torch.Tensor] = None,
                 traffic_controls: Optional[torch.Tensor] = None,
                 lane_actor_index: Optional[torch.Tensor] = None,
                 lane_actor_vectors: Optional[torch.Tensor] = None,
                 seq_id: Optional[int] = None,
                 **kwargs) -> None:
        if x is None:
            super(TemporalData, self).__init__()
            return
        super(TemporalData, self).__init__(x=x, positions=positions, edge_index=edge_index, y=y, num_nodes=num_nodes,
                                           padding_mask=padding_mask, bos_mask=bos_mask, rotate_angles=rotate_angles,
                                           lane_vectors=lane_vectors, is_intersections=is_intersections,
                                           turn_directions=turn_directions, traffic_controls=traffic_controls,
                                           lane_actor_index=lane_actor_index, lane_actor_vectors=lane_actor_vectors,
                                           seq_id=seq_id, **kwargs)
        if edge_attrs is not None:
            for t in range(self.x.size(1)):
                self[f'edge_attr_{t}'] = edge_attrs[t]

    def __inc__(self, key, value):
        if key == 'lane_actor_index':
            return torch.tensor([[self['lane_vectors'].size(0)], [self.num_nodes]])
        else:
            return super().__inc__(key, value)


class DistanceDropEdge(object):

    def __init__(self, max_distance: Optional[float] = None) -> None:
        self.max_distance = max_distance

    def __call__(self,
                 edge_index: torch.Tensor,
                 edge_attr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.max_distance is None:
            return edge_index, edge_attr
        row, col = edge_index
        mask = torch.norm(edge_attr, p=2, dim=-1) < self.max_distance
        edge_index = torch.stack([row[mask], col[mask]], dim=0)
        edge_attr = edge_attr[mask]
        return edge_index, edge_attr

class V2XDropEdge(object):
    """
    find the matching nodes according to iou
    """

    def __init__(self, iou_threshold = 0.3) -> None:
        self.iou_thresh = iou_threshold

    def __call__(self,
                 edge_index_t: torch.Tensor,
                 v2x_positions: torch.Tensor,
                 v2x_width: torch.Tensor,
                 v2x_height: torch.Tensor,
                 v2x_heading: torch.Tensor,
                 source: torch.Tensor,
                 t: int,
                 ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # get car <-> infra matching nodes pairs at timestep t via iou
        # remove matched infra nodes from the graph for a merged graph
        v2x_width = self.fill_nan(v2x_width)
        v2x_height = self.fill_nan(v2x_height)
        v2x_heading = self.fill_nan(v2x_heading)

        # is_matched_car_infra_nodes = torch.zeros(edge_index_t.shape[1], dtype=torch.bool)
        # Find car-infra pairs
        car_infra_masks = torch.isin(edge_index_t[0], (source == 0).nonzero(as_tuple=True)[0]) & \
                          torch.isin(edge_index_t[1], (source == 1).nonzero(as_tuple=True)[0]) 
        src, tgt = edge_index_t[0,car_infra_masks], edge_index_t[1, car_infra_masks]
        bbox1 = torch.stack((v2x_positions[src,t,0], v2x_positions[src,t,1], 
                             v2x_width[src,t], v2x_height[src,t], v2x_heading[src,t]), dim=1)
        bbox2 = torch.stack((v2x_positions[tgt,t,0], v2x_positions[tgt,t,1], v2x_width[tgt,t], 
                             v2x_height[tgt,t], v2x_heading[tgt,t]), dim=1)
        
        rotated_iou = self.iou(bbox1, bbox2)
        matched_car_infra_nodes = edge_index_t[:,car_infra_masks][:,rotated_iou>self.iou_thresh]
        nodes_to_keep = edge_index_t.unique()[~torch.isin(edge_index_t.unique(), matched_car_infra_nodes[1].unique())]

        return nodes_to_keep, matched_car_infra_nodes

    def iou(self, bbox1, bbox2):
        x1, y1, w1, h1 = bbox1[:,0] - bbox1[:,2]/2, bbox1[:,1] - bbox1[:,3]/2, bbox1[:,2], bbox1[:,3]
        x2, y2, w2, h2 = bbox2[:,0] - bbox2[:,2]/2, bbox2[:,1] - bbox2[:,3]/2, bbox2[:,2], bbox2[:,3]

        overlap_left = torch.max(x1, x2)
        overlap_right = torch.min(x1+w1, x2+w2)
        overlap_top = torch.max(y1, y2)
        overlap_bottom = torch.min(y1+h1, y2+h2)
        # overlap_w = torch.max(torch.zeros(*overlap_right.shape, device=self.device), overlap_right - overlap_left)
        # overlap_h = torch.max(torch.zeros(*overlap_bottom.shape, device=self.device), overlap_bottom - overlap_top)
        overlap_w = torch.clamp(overlap_right - overlap_left, min=0.)
        overlap_h = torch.clamp(overlap_bottom - overlap_top, min=0.)
        overlap_area = overlap_w * overlap_h
        area_1 = w1 * h1
        area_2 = w2 * h2

        iou = overlap_area / (area_1 + area_2 - overlap_area)

        return iou
    def fill_nan(self, input):
        nan_mask = torch.isnan(input)
        median = torch.median(input[~nan_mask])
        input[nan_mask] = median
        return input

def init_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        fan_in = m.in_channels / m.groups
        fan_out = m.out_channels / m.groups
        bound = (6.0 / (fan_in + fan_out)) ** 0.5
        nn.init.uniform_(m.weight, -bound, bound)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.MultiheadAttention):
        if m.in_proj_weight is not None:
            fan_in = m.embed_dim
            fan_out = m.embed_dim
            bound = (6.0 / (fan_in + fan_out)) ** 0.5
            nn.init.uniform_(m.in_proj_weight, -bound, bound)
        else:
            nn.init.xavier_uniform_(m.q_proj_weight)
            nn.init.xavier_uniform_(m.k_proj_weight)
            nn.init.xavier_uniform_(m.v_proj_weight)
        if m.in_proj_bias is not None:
            nn.init.zeros_(m.in_proj_bias)
        nn.init.xavier_uniform_(m.out_proj.weight)
        if m.out_proj.bias is not None:
            nn.init.zeros_(m.out_proj.bias)
        if m.bias_k is not None:
            nn.init.normal_(m.bias_k, mean=0.0, std=0.02)
        if m.bias_v is not None:
            nn.init.normal_(m.bias_v, mean=0.0, std=0.02)
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                for ih in param.chunk(4, 0):
                    nn.init.xavier_uniform_(ih)
            elif 'weight_hh' in name:
                for hh in param.chunk(4, 0):
                    nn.init.orthogonal_(hh)
            elif 'weight_hr' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias_ih' in name:
                nn.init.zeros_(param)
            elif 'bias_hh' in name:
                nn.init.zeros_(param)
                nn.init.ones_(param.chunk(4, 0)[1])
    elif isinstance(m, nn.GRU):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                for ih in param.chunk(3, 0):
                    nn.init.xavier_uniform_(ih)
            elif 'weight_hh' in name:
                for hh in param.chunk(3, 0):
                    nn.init.orthogonal_(hh)
            elif 'bias_ih' in name:
                nn.init.zeros_(param)
            elif 'bias_hh' in name:
                nn.init.zeros_(param)
