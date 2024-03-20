import torch
import torch.nn as nn
import torch.nn.functional as F

from coop_models.modelBase import TemporalEncoder, AA_GAT, cross_graph_attention, ALEncoder, PredictionDecoder
from dataloader.temporal_data import TemporalData, get_lane_features
from coop_models.utils import MLP, DistanceDropEdge

from torch_geometric.utils import subgraph
import torchvision

from dataset.dair_map_api import DAIRV2XMap


class CoopModel(nn.Module):
    def __init__(self,
                 ip_dim: int=2,
                 historical_steps: int=50,
                 embed_dim: int=16,
                 num_heads: int=8,
                 num_temporal_layers: int=4,
                 dropout: float=0.1,
                 model_radius: float=50,
                 device = "cpu",
                 use_variance: bool = True):
        super(CoopModel, self).__init__()

        self.historical_steps = historical_steps
        self.model_radius = model_radius
        self.device = device
        # self.ip_emb = MLP(ip_dim, embed_dim)
        self.infra_graph = AA_GAT(node_dim = ip_dim, 
                                  embed_dim = embed_dim, 
                                  out_dim = embed_dim, 
                                  edge_attr_dim = ip_dim, 
                                  device = device, 
                                  num_heads=num_heads, 
                                  dropout=dropout)
        self.car_graph = AA_GAT(node_dim = ip_dim, 
                                  embed_dim = embed_dim, 
                                  out_dim = embed_dim, 
                                  edge_attr_dim = ip_dim, 
                                  device = device, 
                                  num_heads=num_heads, 
                                  dropout=dropout)
        self.cross_graph_attention = cross_graph_attention(embed_dim=embed_dim*num_heads,
                                                           num_heads=num_heads,
                                                           dropout=dropout)
        # self.infra_temp_encoder = TemporalEncoder(historical_steps=historical_steps,
        #                                     embed_dim=embed_dim,
        #                                     num_heads=num_heads,
        #                                     num_layers=num_temporal_layers,
        #                                     dropout=dropout)
        # self.car_temp_encoder = TemporalEncoder(historical_steps=historical_steps,
        #                                     embed_dim=embed_dim,
        #                                     num_heads=num_heads,
        #                                     num_layers=num_temporal_layers,
        #                                     dropout=dropout)
        self.temp_encoder = TemporalEncoder(historical_steps=historical_steps,
                                    embed_dim=embed_dim*num_heads,
                                    num_heads=num_heads,
                                    num_layers=num_temporal_layers,
                                    dropout=dropout)
        self.dm = DAIRV2XMap()
        self.al_encoder = ALEncoder(node_dim=ip_dim,
                                    edge_dim=ip_dim,
                                    embed_dim=embed_dim*num_heads,
                                    num_heads=num_heads,
                                    dropout=dropout)
        self.decoder = PredictionDecoder(encoding_size=embed_dim*num_heads,
                                         hidden_size=64,
                                         num_modes=5,
                                         op_len=50,
                                         use_variance=use_variance)
        self.drop_edge = DistanceDropEdge(model_radius)

    def forward(self, infra_data: TemporalData, car_data: TemporalData, graph_match_matrix):
        # upon receiving two datasets, transform global coordinates into AV centered coords and vectorize.
        # infra_data, car_data, rotate_mat, rotate_imat = self.get_transformed_vectorized_coords(infra_data, car_data)
        #build AV and RSU graphs separately at each timestamp to aggregate neighboring informations
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
        #fuse information from two graphs according to iou_matrix at each timestamp
        fused_graph = [None] * self.historical_steps
        valid_mask = [None] * self.historical_steps
        for t in range(self.historical_steps):
            valid_cars_t = ~car_data["padding_mask"][:,t] #nonzero coordinates
            valid_infra_t = ~infra_data["padding_mask"][:,t]
            graph_match_matrix_t = self.get_iou_mat(infra_data, car_data, t, valid_cars_t, valid_infra_t)

            infra_data[f'edge_index_{t}'], _ = subgraph(subset=valid_infra_t, edge_index=infra_data.edge_index)
            infra_data[f'edge_attr_{t}'] = \
                infra_data['positions'][infra_data[f'edge_index_{t}'][0], t] - infra_data['positions'][infra_data[f'edge_index_{t}'][1], t]
            
            car_data[f'edge_index_{t}'], _ = subgraph(subset=valid_cars_t, edge_index=car_data.edge_index)
            car_data[f'edge_attr_{t}'] = \
                car_data['positions'][car_data[f'edge_index_{t}'][0], t] - car_data['positions'][car_data[f'edge_index_{t}'][1], t]

            # get infra graph encoding
            infra_edge_index_t, infra_edge_attr_t = self.drop_edge(infra_data[f'edge_index_{t}'], infra_data[f'edge_attr_{t}'])
            infra_graph_t = self.infra_graph(X=infra_data.x[:, t], edge_index=infra_edge_index_t, edge_attr=infra_edge_attr_t) #[N1, embed_size]
                                        # bos_mask=infra_data['bos_mask'][:, t], rotate_mat=rotate_imat)
            
            # get AV graph encoding
            car_edge_index_t, car_edge_attr_t = self.drop_edge(car_data[f'edge_index_{t}'], car_data[f'edge_attr_{t}'])
            car_graph_t = self.car_graph(X=car_data.x[:, t], edge_index=car_edge_index_t, edge_attr=car_edge_attr_t) #[N2, embed_size]
                                        # bos_mask=car_data['bos_mask'][:, t], rotate_mat=rotate_imat) 
            
            fused_graph[t], valid_mask[t] = self.cross_graph_attention(infra_graph_t, car_graph_t, graph_match_matrix_t, valid_cars_t, valid_infra_t)

            #use mask for overlapping agents
        fused_graph = torch.stack(fused_graph)  # [T, N, D]
        valid_mask = torch.stack(valid_mask)
        # #temporal encoding
        temp_out = self.temp_encoder(fused_graph, valid_mask) #[N, D]

        #AL encoding
        #recalculate lane_actor graph using all actors' info
        # get lane features at the current time step

        #get agent node index which has appreared in the past 10 frames 
        node_inds_49 = torch.where(valid_mask[-10:,:].sum(dim=0)!=0)[0]
        node_positions_49 = torch.cat([car_data.positions[:,49,:], infra_data.positions[:,49,:]], axis=0)[node_inds_49].double()

        (lane_vectors, is_intersections, turn_directions, traffic_controls, lane_actor_index,
        lane_actor_vectors) = get_lane_features(self.dm, node_inds_49, node_positions_49, car_data.origin, rotate_mat, "PEK", self.model_radius)

        al_edge_index, al_edge_attr = self.drop_edge(lane_actor_index, lane_actor_vectors)
        al_out = self.al_encoder(x=(lane_vectors, temp_out), edge_index=al_edge_index, edge_attr=al_edge_attr,
                              is_intersections=is_intersections, turn_directions=turn_directions,
                              traffic_controls=traffic_controls, rotate_mat=rotate_imat)

        # infra_x, infra_padding_mask = infra_data.x, infra_data.padding_mask[:, :self.historical_steps]
        # car_x, car_padding_mask = car_data.x, car_data.padding_mask[:, :self.historical_steps]
        
        # infra_x_ = torch.transpose(self.ip_emb(infra_x), 1, 0)
        # car_x_ = torch.transpose(self.ip_emb(car_x), 1, 0)

        # infra_temp_out = self.infra_temp_encoder(infra_x_, infra_padding_mask)
        # car_temp_out = self.car_temp_encoder(car_x_, car_padding_mask)

        #fusion
        #model graphs at each timestamp can help eliminate occulusion
        #fuse two edge graphs?
        #then temporal encoding

        #decoder
        out = self.decoder(al_out) #traj, log_probs
        out = al_out
        # infra_predictions = self.decoder(infra_temp_out)
        # car_predictions = self.decoder(car_temp_out)


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
    
    # import matplotlib.pyplot as plt
    # pred = infra_predictions["traj"][0]
    # for i in range(5):
    #      plt.plot(pred.detach().numpy()[i,:,0], pred.detach().numpy()[i,:,1])
    # plt.show()
# for i in range(car_data.positions.shape[0]):           
#     plt.plot(car_data.positions[i,:,0][~car_data.padding_mask[i,:]], car_data.positions[i,:,1][~car_data.padding_mask[i,:]])

# for i in range(test0.shape[0]):           
#     plt.plot(test0[i,:,0][~car_data.padding_mask[i,:]], test0[i,:,1][~car_data.padding_mask[i,:]])