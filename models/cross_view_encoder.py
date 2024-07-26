from typing import Optional

import torch
import torch.nn as nn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, Tuple, OptTensor, Size
from torch_geometric.utils import softmax
from torch_geometric.utils import subgraph, bipartite_subgraph

from models import MultipleInputEmbedding
from utils import TemporalData
from utils import init_weights


class CrossViewEncoder(nn.Module):

    def __init__(self,
                 historical_steps: int,
                 embed_dim: int,
                 edge_dim: int,
                 num_modes: int = 6,
                 num_heads: int = 8,
                 num_layers: int = 4,
                 dropout: float = 0.1) -> None:
        super(CrossViewEncoder, self).__init__()
        self.historical_steps = historical_steps
        self.embed_dim = embed_dim
        self.num_modes = num_modes

        self.rel_embed = MultipleInputEmbedding(in_channels=[edge_dim, edge_dim], out_channel=embed_dim)
        self.graph_cross_view_layers = nn.ModuleList(
            [GlobalCrossViewLayer(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
             for _ in range(num_layers)])
        self.norm = nn.LayerNorm(embed_dim)
        self.multihead_proj = nn.Linear(embed_dim, num_modes * embed_dim)
        self.apply(init_weights)

    def forward(self,
                data: TemporalData,
                car_view_embed: torch.Tensor,
                infra_view_embed: torch.Tensor) -> torch.Tensor:
        source_mask_car, source_mask_infra = data.source==1, data.source==0        
        infra_car_index, _ = bipartite_subgraph(subset=(source_mask_infra, source_mask_car), edge_index=data.edge_index)
        edge_index, _ = subgraph(subset=~data['padding_mask'][:, self.historical_steps - 1], edge_index=infra_car_index)
        rel_pos = data['positions'][edge_index[0], self.historical_steps - 1] - data['positions'][
            edge_index[1], self.historical_steps - 1]

        rel_pos = torch.bmm(rel_pos.unsqueeze(-2), data['rotate_mat'][edge_index[1]]).squeeze(-2)
        rel_theta = data['rotate_angles'][edge_index[0]] - data['rotate_angles'][edge_index[1]]
        rel_theta_cos = torch.cos(rel_theta).unsqueeze(-1)
        rel_theta_sin = torch.sin(rel_theta).unsqueeze(-1)
        rel_embed = self.rel_embed([rel_pos, torch.cat((rel_theta_cos, rel_theta_sin), dim=-1)])

        x_infra, x_car = infra_view_embed, car_view_embed
        for layer in self.graph_cross_view_layers:
            x_infra, x_car = layer((x_infra, x_car), edge_index, rel_embed)
        x = self.norm(x_car)  # [N, D]
        x = self.multihead_proj(x).view(-1, self.num_modes, self.embed_dim)  # [N, F, D]
        x = x.transpose(0, 1)  # [F, N, D]
        return x


class GlobalCrossViewLayer(MessagePassing):

    def __init__(self,
                 embed_dim: int,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 **kwargs) -> None:
        super(GlobalCrossViewLayer, self).__init__(aggr='add', node_dim=0, **kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.lin_q_node = nn.Linear(embed_dim, embed_dim)
        self.lin_k_node = nn.Linear(embed_dim, embed_dim)
        self.lin_k_edge = nn.Linear(embed_dim, embed_dim)
        self.lin_v_node = nn.Linear(embed_dim, embed_dim)
        self.lin_v_edge = nn.Linear(embed_dim, embed_dim)
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

    def forward(
            self,
            x: Tuple[torch.Tensor, torch.Tensor],
            edge_index: Adj,
            edge_attr: torch.Tensor,
            size: Size = None) -> torch.Tensor:
        
        x_infra, x_car = x
        x_car = x_car + self._mha_block(self.norm1(x_car), x_infra, edge_index, edge_attr, size)
        x_car = x_car + self._ff_block(self.norm2(x_car))

        return x_infra, x_car

    def message(self,
                x_i: torch.Tensor,
                x_j: torch.Tensor,
                edge_attr: torch.Tensor,
                index: torch.Tensor,
                ptr: OptTensor,
                size_i: Optional[int]) -> torch.Tensor:
        query = self.lin_q_node(x_i).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        key_node = self.lin_k_node(x_j).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        key_edge = self.lin_k_edge(edge_attr).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        value_node = self.lin_v_node(x_j).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        value_edge = self.lin_v_edge(edge_attr).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        scale = (self.embed_dim // self.num_heads) ** 0.5
        alpha = (query * (key_node + key_edge)).sum(dim=-1) / scale
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = self.attn_drop(alpha)
        return (value_node + value_edge) * alpha.unsqueeze(-1)

    def update(self,
               inputs: torch.Tensor,
               x: torch.Tensor) -> torch.Tensor:
        x_car = x[1]
        inputs = inputs.view(-1, self.embed_dim)
        gate = torch.sigmoid(self.lin_ih(inputs) + self.lin_hh(x_car))
        return inputs + gate * (self.lin_self(x_car) - inputs)

    def _mha_block(self,
                   x_car: torch.Tensor,
                   x_infra: torch.Tensor,
                   edge_index: Adj,
                   edge_attr: torch.Tensor,
                   size: Size) -> torch.Tensor:
        x_car = self.out_proj(self.propagate(edge_index=edge_index, x=(x_infra, x_car), edge_attr=edge_attr,
                                               size=size))
        return self.proj_drop(x_car)

    def _ff_block(self, x_car: torch.Tensor) -> torch.Tensor:
        return self.mlp(x_car)