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
from torch_geometric.utils import subgraph, add_self_loops
    
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
                 dropout: float = 0.1,
                 temp_ff: int = 128) -> None:
        super(TemporalEncoder, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(embed_dim, num_heads, dim_feedforward=temp_ff, dropout=dropout)
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
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self,
                x: torch.Tensor,
                padding_mask: torch.Tensor) -> torch.Tensor:
        """ x [seq, batch, feature]"""
        x = torch.where(padding_mask.t().unsqueeze(-1), self.padding_token, x)
        expand_cls_token = self.cls_token.expand(-1, x.shape[1], -1)
        x = torch.cat((x, expand_cls_token), dim=0)
        x = x + self.pos_embed
        x = self.layer_norm(self.dropout(x)) #dropout and layernorm
        out = self.transformer_encoder(src=x, mask=self.attn_mask, src_key_padding_mask=None)
        return out[-1]  # [N, D]

    @staticmethod
    def generate_square_subsequent_mask(seq_len: int) -> torch.Tensor:
        mask = (torch.triu(torch.ones(seq_len, seq_len)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
class ALEncoder(nn.Module):
    def __init__(self,
                 lane_dim: int,
                 out_dim: int,
                 edge_attr_dim: int,
                 num_heads: int,
                 local_radius: float=30.,
                 dropout: float=0.1) -> None:
        super(ALEncoder, self).__init__()
        self.local_radius = local_radius
        self.attention_layers = nn.ModuleList(
            [ALEncoderLayer(out_dim, out_dim, edge_attr_dim) for _ in range(num_heads)]
        )
        self.lane_emb = MLP(lane_dim, out_dim) #out_dim = v_enc.size(1)
        self.edge_attr_dim = edge_attr_dim
        self.dropout = nn.Dropout(dropout)
        self.out_transform = nn.Linear(out_dim*num_heads, out_dim, bias=False)

    def forward(self, lane_vectors: torch.Tensor, 
                lane_actor_index: torch.Tensor, 
                num_nodes: torch.Tensor,
                rotate_imat: torch.Tensor,  
                lane_actor_vectors: torch.Tensor, 
                v_enc: torch.Tensor):

        lane = lane_vectors

        # lane_actor_mask = torch.cat((v_mask, (torch.ones(lane.size(0))==1).to(self.device)), dim=0)
        lane_actor_index[0] += num_nodes #lane_actor_index[0]:lane index, lane_actor_index[1]:actor index
        # lane_actor_index, lane_actor_attr = subgraph(subset=lane_actor_mask, 
        #                                edge_index=data.lane_actor_index, edge_attr=data.lane_actor_attr)
        lane = torch.bmm(lane[lane_actor_index[0]-num_nodes].unsqueeze(-2), rotate_imat[lane_actor_index[1]].float()).squeeze(-2)

        lane_enc = self.lane_emb(lane)
        lane_actor_enc = torch.cat((v_enc, lane_enc), dim=0) #shape:[num_veh+num_lane, v_dim]
        # Concat multi-head attentions
        out = torch.cat([att(lane_actor_enc, num_nodes, lane.size(0), lane_actor_index, lane_actor_vectors) for att in self.attention_layers], dim=1) 
        out = F.elu(out)
        out = self.dropout(out)
        out = self.out_transform(out)

        return out
    
class ALEncoderLayer(nn.Module):
    def __init__(self,
                 v_dim: int,
                 out_dim: int,
                 edge_attr_dim: int,
                 dropout: float=0.1) -> None:
        super(ALEncoderLayer, self).__init__()
        
        self.W = nn.Linear(v_dim, out_dim, bias=False)
        self.a = nn.Linear(2*out_dim + edge_attr_dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, 
                lane_actor_enc: torch.Tensor, 
                num_veh: int, 
                num_lane: int,
                lane_actor_index: torch.Tensor,
                lane_actor_attr: torch.Tensor):
        #transform node features
        h = self.W(lane_actor_enc)
        N = h.size(0)
        assert N == num_veh+num_lane

        attn_input = self._prepare_attention_input(h, num_veh,lane_actor_index, lane_actor_attr)
        score_per_edge = F.leaky_relu(self.a(attn_input)).squeeze(1)  # Calculate attention coefficients
        
        #apply dropout to attention weights
        score_per_edge = self.dropout(score_per_edge)
        # softmax
        # Calculate the numerator. Make logits <= 0 so that e^logit <= 1 (this will improve the numerical stability)
        score_per_edge = score_per_edge - score_per_edge.max()
        exp_score_per_edge = score_per_edge.exp()  

        neigborhood_aware_denominator = scatter_add(exp_score_per_edge, lane_actor_index[1], dim=0, dim_size=num_veh)
        neigborhood_aware_denominator = neigborhood_aware_denominator.index_select(0, lane_actor_index[1])
        attentions_per_edge = exp_score_per_edge / (neigborhood_aware_denominator + 1e-16)

        out_src = h[num_veh:] * attentions_per_edge.unsqueeze(dim=1) #shape[num_lane]
        out = scatter_add(out_src, lane_actor_index[1], dim=0, dim_size=num_veh)
        assert out.shape[0] == num_veh

        # Apply activation function
        out = F.elu(out)
        return out
    
    def _prepare_attention_input(self, h, num_v, edge_index, edge_attr):
        '''
        h has shape [N, out_dim]
        '''
        src, tgt = edge_index
        attn_input = torch.cat([h[num_v:], h[:num_v].index_select(0,tgt),  edge_attr], dim=1)

        return attn_input   


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
