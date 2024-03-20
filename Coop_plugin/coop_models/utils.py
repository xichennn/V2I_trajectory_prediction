from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops

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

def bivariate_gaussian_activation(ip: torch.Tensor) -> torch.Tensor:
    """
    Activation function to output parameters of bivariate Gaussian distribution
    """
    mu_x = ip[..., 0:1]
    mu_y = ip[..., 1:2]
    sig_x = ip[..., 2:3]
    sig_y = ip[..., 3:4]
    rho = ip[..., 4:5]
    sig_x = torch.exp(sig_x)
    sig_y = torch.exp(sig_y)
    rho = torch.tanh(rho)
    out = torch.cat([mu_x, mu_y, sig_x, sig_y, rho], dim = -1)

    return out

class MLP(nn.Module):
    def __init__(self,
                 in_dim: int, 
                 out_dim: int) -> None:
        super(MLP, self).__init__()
        self.embed = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim)
        )
        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embed(x)
    
def iou(bbox1,bbox2):
    """
    Calculates the intersection-over-union of two bounding boxes.
    Args:
        bbox1 (numpy.array, list of floats): bounding box in format x1,y1,x2,y2.
        bbox2 (numpy.array, list of floats): bounding box in format x1,y1,x2,y2.
    Returns:
        int: intersection-over-onion of bbox1, bbox2
    """
    bbox1 = [float(x) for x in bbox1]
    bbox2 = [float(x) for x in bbox2]

    (x0_1, y0_1, x1_1, y1_1) = bbox1
    (x0_2, y0_2, x1_2, y1_2) = bbox2

    # get the overlap rectangle
    overlap_x0 = max(x0_1, x0_2)
    overlap_y0 = max(y0_1, y0_2)
    overlap_x1 = min(x1_1, x1_2)
    overlap_y1 = min(y1_1, y1_2)

    # check if there is an overlap
    if overlap_x1 - overlap_x0 <= 0 or overlap_y1 - overlap_y0 <= 0:
        return 0

    # if yes, calculate the ratio of the overlap to each ROI size and the unified size
    size_1 = (x1_1 - x0_1) * (y1_1 - y0_1)
    size_2 = (x1_2 - x0_2) * (y1_2 - y0_2)
    size_intersection = (overlap_x1 - overlap_x0) * (overlap_y1 - overlap_y0)
    size_union = size_1 + size_2 - size_intersection

    return size_intersection / size_union

def cal_box_iou_centerpts(centerpoints1, w1, h1,centerpoints2, w2, h2):
    ct_x1, ct_y1 = centerpoints1[0], centerpoints1[1]
    ct_x2, ct_y2 = centerpoints2[0], centerpoints2[1]

    box1 = [ct_x1 - w1 / 2, ct_y1 - h1 / 2, ct_x1 + w1 / 2, ct_y1 + h1 / 2]
    box2 = [ct_x2 - w2 / 2, ct_y2 - h2 / 2, ct_x2 + w2 / 2, ct_y2 + h2 / 2]

    return iou(box1,box2)

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
                 edge_attr_t: torch.Tensor,
                 v2x_positions: torch.Tensor,
                 v2x_width: torch.Tensor,
                 v2x_height: torch.Tensor,
                 source: torch.Tensor,
                 t: int,
                 ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # get car <-> infra matching nodes pairs at timestep t via iou
        # remove matched infra nodes from the graph for a merged graph

        # v2x_positions = v2x_data["positions"]
        # v2x_width = v2x_data["width"]
        # v2x_height = v2x_data["height"]
        # edge_index_t = v2x_data[f'edge_index_{t}']
        # edge_attr_t = v2x_data[f'edge_attr_{t}']
        # source = v2x_data["source"]
        edge0_bbox_t = torch.stack((v2x_positions[edge_index_t[0],t,0] - v2x_width[edge_index_t[0],t]/2, 
                                v2x_positions[edge_index_t[0],t,1] - v2x_height[edge_index_t[0],t]/2,
                                v2x_positions[edge_index_t[0],t,0] + v2x_width[edge_index_t[0],t]/2, 
                                v2x_positions[edge_index_t[0],t,1] + v2x_height[edge_index_t[0],t]/2), dim=1)
        edge1_bbox_t = torch.stack((v2x_positions[edge_index_t[1],t,0] - v2x_width[edge_index_t[1],t]/2, 
                                v2x_positions[edge_index_t[1],t,1] - v2x_height[edge_index_t[1],t]/2,
                                v2x_positions[edge_index_t[1],t,0] + v2x_width[edge_index_t[1],t]/2, 
                                v2x_positions[edge_index_t[1],t,1] + v2x_height[edge_index_t[1],t]/2), dim=1)
        #find out edge_index for car-infra pairs
        is_matched_car_infra_nodes = torch.zeros(edge_index_t.shape[1], dtype=torch.bool)
        for i in range(edge_index_t.shape[1]):
            src, tgt = edge_index_t[0,i], edge_index_t[1,i]
            if source[src]==0 and source[tgt]==1: #car_infra pair
                if iou(edge0_bbox_t[i], edge1_bbox_t[i]) > self.iou_thresh:
                    is_matched_car_infra_nodes[i] = True
        matched_car_infra_nodes = edge_index_t[:,is_matched_car_infra_nodes]


        # iou_t = torch.zeros(edge0_bbox_t.shape[0])
        # for i in range(edge0_bbox_t.shape[0]):
        #     iou_t[i] = iou(edge0_bbox_t[i], edge1_bbox_t[i])
        
        # matched_edge_index = edge_index_t[:,iou_t > self.iou_thresh]
        # matched_car_infra_nodes_indices = [i for i in range(matched_edge_index.shape[1]) 
        #                             if source[matched_edge_index[0,i]]== 1 and source[matched_edge_index[1,i]]== 0]
        # matched_car_infra_nodes = matched_edge_index[:,matched_car_infra_nodes_indices]
        # matched_infra_nodes = matched_car_infra_nodes[1]
        # # matched_infra_nodes = [int(i) for i in matched_edge_index[0,:] if source[i]== 0] #source0: infra
        # edge_mask0 = torch.any(edge_index_t[0].unsqueeze(0) == torch.Tensor(matched_infra_nodes).view(-1, 1), dim=0)
        # edge_mask1 = torch.any(edge_index_t[1].unsqueeze(0) == torch.Tensor(matched_infra_nodes).view(-1, 1), dim=0)
        # edge_mask = edge_mask0 | edge_mask1

        # merged_edge_index = edge_index_t[:,~edge_mask]
        merged_edge_index, merged_edge_attr = self.drop_edges_given_nodes(matched_car_infra_nodes[1], edge_index_t, edge_attr_t)

        # matched_edge_attr = edge_attr_t[iou_t > self.iou_thresh]
        # merged_edge_attr = edge_attr_t[~edge_mask]

        return merged_edge_index, merged_edge_attr, matched_car_infra_nodes
    def drop_edges_given_nodes(self, dropped_nodes, edge_index, edge_attr):
        
        # matched_infra_nodes = [int(i) for i in matched_edge_index[0,:] if source[i]== 0] #source0: infra
        edge_mask0 = torch.any(edge_index[0].unsqueeze(0) == torch.Tensor(dropped_nodes).view(-1, 1), dim=0)
        edge_mask1 = torch.any(edge_index[1].unsqueeze(0) == torch.Tensor(dropped_nodes).view(-1, 1), dim=0)
        edge_mask = edge_mask0 | edge_mask1

        edge_index_dropped, edge_attr_dropped = edge_index[:,~edge_mask], edge_attr[~edge_mask]

        return edge_index_dropped, edge_attr_dropped
    
class MultipleInputEmbedding(nn.Module):

    def __init__(self,
                 in_channels: List[int],
                 out_channel: int) -> None:
        super(MultipleInputEmbedding, self).__init__()
        self.module_list = nn.ModuleList(
            [nn.Sequential(nn.Linear(in_channel, out_channel),
                           nn.LayerNorm(out_channel),
                           nn.ReLU(inplace=True),
                           nn.Linear(out_channel, out_channel))
             for in_channel in in_channels])
        self.aggr_embed = nn.Sequential(
            nn.LayerNorm(out_channel),
            nn.ReLU(inplace=True),
            nn.Linear(out_channel, out_channel),
            nn.LayerNorm(out_channel))
        self.apply(init_weights)

    def forward(self,
                continuous_inputs: List[torch.Tensor],
                categorical_inputs: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        for i in range(len(self.module_list)):
            continuous_inputs[i] = self.module_list[i](continuous_inputs[i])
        output = torch.stack(continuous_inputs).sum(dim=0)
        if categorical_inputs is not None:
            output += torch.stack(categorical_inputs).sum(dim=0)
        return self.aggr_embed(output)