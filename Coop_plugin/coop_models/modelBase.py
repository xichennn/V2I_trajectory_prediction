from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .utils import init_weights, bivariate_gaussian_activation, MultipleInputEmbedding
from torch_scatter import scatter_mean, scatter_add
from coop_models.utils import MLP

from torch_geometric.typing import Adj
from torch_geometric.typing import OptTensor
from torch_geometric.typing import Size
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax
    
class AA_GAT(nn.Module):
    def __init__(self, node_dim, embed_dim, out_dim, edge_attr_dim, device, num_heads=8, dropout=0.1):
        super(AA_GAT, self).__init__()

        self.device = device
        self.node_embed = MLP(node_dim, embed_dim)
        self.edge_attr_embed = MLP(edge_attr_dim, embed_dim)
        self.attention_layers = nn.ModuleList(
            [AA_GATlayer(embed_dim, out_dim, embed_dim) for _ in range(num_heads)]
        )
        self.out_att = AA_GATlayer(out_dim*num_heads, out_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.cross_att = cross_graph_attention(embed_dim, num_heads)

    def forward(self, X, edge_index, edge_attr, matched_car_infra_nodes):

        x = self.node_embed(X)
        # x[matched_car_infra_nodes[0]] = self.cross_att(x[matched_car_infra_nodes[0]], x[matched_car_infra_nodes[1]])
        edge_attr_embed = self.edge_attr_embed(edge_attr)
        
        # Concatenate multi-head attentions
        x = torch.cat([att(x, edge_index, edge_attr_embed) for att in self.attention_layers], dim=1) 
        x = F.elu(x)
        x = self.dropout(x)
        x = self.out_att(x, edge_index, edge_attr_embed) # Final attention aggregation
        return F.log_softmax(x, dim=1)
    
class AA_GATlayer(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 out_dim: int,
                 edge_attr_dim: int,
                 dropout: float=0.1) -> None:
        super(AA_GATlayer, self).__init__()
        
        self.W = nn.Linear(embed_dim, out_dim, bias=False)
        self.a = nn.Linear(2*out_dim + edge_attr_dim, 1, bias=False)
        self.edge_attr_dim = edge_attr_dim
        self.dropout = nn.Dropout(dropout)
        self.out_transform = nn.Linear(out_dim, out_dim, bias=False)

    def forward(self, 
                X: torch.Tensor, 
                edge_index: torch.Tensor, 
                edge_attr: torch.Tensor):
        #transform node features
        h = self.W(X)
        N = h.size(0)
        attn_input = self._prepare_attention_input(h, edge_index, edge_attr)
        score_per_edge = F.leaky_relu(self.a(attn_input)).squeeze(1)  # Calculate attention coefficients
        
        #apply dropout to attention weights
        score_per_edge = self.dropout(score_per_edge)
        # softmax
        # Calculate the numerator. Make logits <= 0 so that e^logit <= 1 (this will improve the numerical stability)
        score_per_edge = score_per_edge - score_per_edge.max()
        exp_score_per_edge = score_per_edge.exp()  

        neigborhood_aware_denominator = scatter_add(exp_score_per_edge, edge_index[0], dim=0, dim_size=N)
        neigborhood_aware_denominator = neigborhood_aware_denominator.index_select(0, edge_index[0])
        attentions_per_edge = exp_score_per_edge / (neigborhood_aware_denominator + 1e-16)
        
        # Apply attention weights to source node features and perform message passing
        out_src = h.index_select(0,edge_index[1]) * attentions_per_edge.unsqueeze(dim=1)
        h_prime = scatter_add(out_src, edge_index[0], dim=0, dim_size=N)

        # Apply activation function
        out = F.elu(h_prime)
        return out

    def _prepare_attention_input(self, h, edge_index, edge_attr):
        '''
        h has shape [N, out_dim]
        '''
        src, tgt = edge_index
        attn_input = torch.cat([h.index_select(0,src), h.index_select(0,tgt),  edge_attr], dim=1)

        return attn_input
    
class cross_graph_attention(nn.Module):
    def __init__(self,
                 embed_dim,
                 num_heads,
                 dropout=0.1):
        super(cross_graph_attention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.lin_q = nn.Linear(embed_dim, embed_dim)
        self.lin_k = nn.Linear(embed_dim, embed_dim)
        self.lin_v = nn.Linear(embed_dim, embed_dim)
        self.lin_self = nn.Linear(embed_dim, embed_dim)
        self.lin_ih = nn.Linear(embed_dim, embed_dim)
        self.lin_hh = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, car_feat, infra_feat):
        query = self.lin_q(car_feat).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        key = self.lin_k(infra_feat).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        value = self.lin_v(infra_feat).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        scale = (self.embed_dim // self.num_heads) ** 0.5
        alpha = (query * key).sum(dim=-1) / scale
        alpha = self.softmax(alpha)
        alpha = self.attn_drop(alpha)
        infra_att = (value * alpha.unsqueeze(-1)).reshape(-1, self.embed_dim)
        w = torch.sigmoid(self.lin_ih(car_feat) + self.lin_hh(infra_att))
        fused_feat = w * self.lin_self(car_feat) + (1-w) * infra_att
        return fused_feat
    
class TemporalEncoder(nn.Module):

    def __init__(self,
                 historical_steps: int,
                 embed_dim: int,
                 num_heads: int = 8,
                 num_layers: int = 4,
                 dropout: float = 0.1) -> None:
        super(TemporalEncoder, self).__init__()
        encoder_layer = TemporalEncoderLayer(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers,
                                                         norm=nn.LayerNorm(embed_dim))
        self.padding_token = nn.Parameter(torch.Tensor(historical_steps, 1, embed_dim))
        self.cls_token = nn.Parameter(torch.Tensor(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.Tensor(historical_steps + 1, 1, embed_dim))
        attn_mask = self.generate_square_subsequent_mask(historical_steps + 1)
        self.register_buffer('attn_mask', attn_mask)
        nn.init.normal_(self.padding_token, mean=0., std=.02)
        nn.init.normal_(self.cls_token, mean=0., std=.02)
        nn.init.normal_(self.pos_embed, mean=0., std=.02)
        self.apply(init_weights)

    def forward(self,
                x: torch.Tensor,
                padding_mask: torch.Tensor) -> torch.Tensor:
        x = torch.where(padding_mask.t().unsqueeze(-1), self.padding_token, x)
        expand_cls_token = self.cls_token.expand(-1, x.shape[1], -1)
        x = torch.cat((x, expand_cls_token), dim=0)
        x = x + self.pos_embed
        out = self.transformer_encoder(src=x, mask=self.attn_mask, src_key_padding_mask=None)
        return out[-1]  # [N, D]

    @staticmethod
    def generate_square_subsequent_mask(seq_len: int) -> torch.Tensor:
        mask = (torch.triu(torch.ones(seq_len, seq_len)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


class TemporalEncoderLayer(nn.Module):

    def __init__(self,
                 embed_dim: int,
                 num_heads: int = 8,
                 dropout: float = 0.1) -> None:
        super(TemporalEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        self.linear1 = nn.Linear(embed_dim, embed_dim * 4)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(embed_dim * 4, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self,
                src: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None,
                **kwargs) -> torch.Tensor:
        x = src
        x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
        x = x + self._ff_block(self.norm2(x))
        return x

    def _sa_block(self,
                  x: torch.Tensor,
                  attn_mask: Optional[torch.Tensor],
                  key_padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        x = self.self_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)[0]
        return self.dropout1(x)

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear2(self.dropout(F.relu_(self.linear1(x))))
        return self.dropout2(x)
    
class ALEncoder(MessagePassing):

    def __init__(self,
                 node_dim: int,
                 edge_dim: int,
                 embed_dim: int,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 **kwargs) -> None:
        super(ALEncoder, self).__init__(aggr='add', node_dim=0, **kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.lane_embed = MultipleInputEmbedding(in_channels=[node_dim, edge_dim], out_channel=embed_dim)
        self.lin_q = nn.Linear(embed_dim, embed_dim)
        self.lin_k = nn.Linear(embed_dim, embed_dim)
        self.lin_v = nn.Linear(embed_dim, embed_dim)
        self.lin_self = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.lin_ih = nn.Linear(embed_dim, embed_dim)
        self.lin_hh = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout))
        self.is_intersection_embed = nn.Parameter(torch.Tensor(2, embed_dim))
        self.turn_direction_embed = nn.Parameter(torch.Tensor(4, embed_dim))
        self.traffic_control_embed = nn.Parameter(torch.Tensor(2, embed_dim))
        nn.init.normal_(self.is_intersection_embed, mean=0., std=.02)
        nn.init.normal_(self.turn_direction_embed, mean=0., std=.02)
        nn.init.normal_(self.traffic_control_embed, mean=0., std=.02)
        self.apply(init_weights)

    def forward(self,
                x: Tuple[torch.Tensor, torch.Tensor],
                edge_index: Adj,
                edge_attr: torch.Tensor,
                is_intersections: torch.Tensor,
                turn_directions: torch.Tensor,
                traffic_controls: torch.Tensor,
                rotate_mat: Optional[torch.Tensor] = None,
                size: Size = None) -> torch.Tensor:
        x_lane, x_actor = x
        is_intersections = is_intersections.long()
        turn_directions = turn_directions.long()
        traffic_controls = traffic_controls.long()
        x_actor = x_actor + self._mha_block(self.norm1(x_actor), x_lane, edge_index, edge_attr, is_intersections,
                                            turn_directions, traffic_controls, rotate_mat, size)
        x_actor = x_actor + self._ff_block(self.norm2(x_actor))
        return x_actor

    def message(self,
                edge_index: Adj,
                x_i: torch.Tensor,
                x_j: torch.Tensor,
                edge_attr: torch.Tensor,
                is_intersections_j,
                turn_directions_j,
                traffic_controls_j,
                rotate_mat: Optional[torch.Tensor],
                index: torch.Tensor,
                ptr: OptTensor,
                size_i: Optional[int]) -> torch.Tensor:
        if rotate_mat is None:
            x_j = self.lane_embed([x_j, edge_attr],
                                  [self.is_intersection_embed[is_intersections_j],
                                   self.turn_direction_embed[turn_directions_j],
                                   self.traffic_control_embed[traffic_controls_j]])
        else:
            rotate_mat = rotate_mat[edge_index[1]]
            x_j = self.lane_embed([torch.bmm(x_j.unsqueeze(-2).float(), rotate_mat).squeeze(-2),
                                   torch.bmm(edge_attr.unsqueeze(-2).float(), rotate_mat).squeeze(-2)],
                                  [self.is_intersection_embed[is_intersections_j],
                                   self.turn_direction_embed[turn_directions_j],
                                   self.traffic_control_embed[traffic_controls_j]])
        query = self.lin_q(x_i).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        key = self.lin_k(x_j).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        value = self.lin_v(x_j).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        scale = (self.embed_dim // self.num_heads) ** 0.5
        alpha = (query * key).sum(dim=-1) / scale
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = self.attn_drop(alpha)
        return value * alpha.unsqueeze(-1)

    def update(self,
               inputs: torch.Tensor,
               x: torch.Tensor) -> torch.Tensor:
        x_actor = x[1]
        inputs = inputs.view(-1, self.embed_dim)
        gate = torch.sigmoid(self.lin_ih(inputs) + self.lin_hh(x_actor))
        return inputs + gate * (self.lin_self(x_actor) - inputs)

    def _mha_block(self,
                   x_actor: torch.Tensor,
                   x_lane: torch.Tensor,
                   edge_index: Adj,
                   edge_attr: torch.Tensor,
                   is_intersections: torch.Tensor,
                   turn_directions: torch.Tensor,
                   traffic_controls: torch.Tensor,
                   rotate_mat: Optional[torch.Tensor],
                   size: Size) -> torch.Tensor:
        x_actor = self.out_proj(self.propagate(edge_index=edge_index, x=(x_lane, x_actor), edge_attr=edge_attr,
                                               is_intersections=is_intersections, turn_directions=turn_directions,
                                               traffic_controls=traffic_controls, rotate_mat=rotate_mat, size=size))
        return self.proj_drop(x_actor)

    def _ff_block(self, x_actor: torch.Tensor) -> torch.Tensor:
        return self.mlp(x_actor)


class PredictionDecoder(nn.Module):

    def __init__(self,
                 encoding_size: int,
                 hidden_size: int=64,
                 num_modes: int=5,
                 op_len: int=50,
                 use_variance: bool=False) -> None:
        super(PredictionDecoder, self).__init__()

        self.op_dim = 5 if use_variance else 2
        self.op_len = op_len
        self.num_modes = num_modes
        self.use_variance = use_variance
        self.hidden = nn.Linear(encoding_size, hidden_size)
        self.traj_op = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, self.op_len * self.op_dim * self.num_modes))
        self.prob_op = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, self.num_modes))

        self.leaky_relu = nn.LeakyReLU(0.01)
        self.log_softmax = nn.LogSoftmax(dim=1)


    def forward(self, agg_encoding: torch.Tensor) -> Dict:
        """
        Forward pass for prediction decoder
        :param agg_encoding: aggregated context encoding
        :return predictions: dictionary with 'traj': K predicted trajectories and
            'probs': K corresponding probabilities
        """

        h = self.leaky_relu(self.hidden(agg_encoding))
        num_vehs = h.shape[0] #n_v
        traj = self.traj_op(h) #[n_v, 1250]
        probs = self.log_softmax(self.prob_op(h)) #[n_v, 5]
        traj = traj.reshape(num_vehs, self.num_modes, self.op_len, self.op_dim)
        probs = probs.squeeze(dim=-1)
        traj = bivariate_gaussian_activation(traj) if self.use_variance else traj
        
        predictions = {'traj':traj, 'log_probs':probs}

        return predictions
