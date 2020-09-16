import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from .readout import SumPooling, AvgPooling, MaxPooling
from .norm import Norm

from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder

class ApplyNodeFunc(nn.Module):

    def __init__(self, mlp, norm_type):
        super(ApplyNodeFunc, self).__init__()
        self.mlp = mlp
        self.norm = Norm(norm_type, self.mlp.output_dim)

    def forward(self, graph, h):
        h = self.mlp(graph, h)
        h = self.norm(graph, h)
        return h

class MLP(nn.Module):

    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, norm_type):
        super(MLP, self).__init__()
        self.linear_or_not = True
        self.num_layers = num_layers
        self.output_dim = output_dim
        if num_layers < 1:
            raise ValueError("number of layers should be postive!")
        elif num_layers == 1:
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            self.norm_list = torch.nn.ModuleList()

            for layer in range(num_layers - 1):
                self.norm_list.append(Norm(norm_type, hidden_dim))


    def forward(self, graph, x):
        if self.linear_or_not:
            return self.linear(x)
        else:
            h = x
            for i in range(self.num_layers - 1):
                h = self.linears[i](h)
                h = self.norm_list[i](graph, h)
                h = F.relu(h)

            return self.linears[-1](h)

class GCNConv(nn.Module):
    def __init__(self, apply_func, init_eps=0, learn_eps=False):
        super(GCNConv, self).__init__()
        self.apply_func = apply_func

        self._reducer = fn.mean

        if learn_eps:
            self.eps = torch.nn.Parameter(torch.FloatTensor([init_eps]))
        else:
            self.register_buffer('eps', torch.FloatTensor([init_eps]))


    def forward(self, g, split_list, node_feat, edge_feat):
        graph = g.local_var()
        graph.ndata['h_n'] = node_feat
        graph.edata['h_e'] = edge_feat
        graph.update_all(
            fn.u_add_e('h_n', 'h_e', 'm'),
                         self._reducer('m', 'neigh'))
        rst = (1 + self.eps) * node_feat + graph.ndata['neigh']

        if self.apply_func is not None:
            rst = self.apply_func(g, rst)
        return rst

class GCN(nn.Module):
    def __init__(self, num_layers, hidden_dim,
                 output_dim, final_dropout, learn_eps, graph_pooling_type,
                 norm_type):
        super(GCN, self).__init__()
        self.num_layers = num_layers
        self.learn_eps = learn_eps

        self.ginlayers = torch.nn.ModuleList()
        self.atom_encoder = AtomEncoder(hidden_dim)

        self.bond_layers = torch.nn.ModuleList()

        for layer in range(self.num_layers - 1):

            mlp = MLP(1, hidden_dim, hidden_dim * 2, hidden_dim, norm_type)

            self.ginlayers.append(
                GCNConv(ApplyNodeFunc(mlp, norm_type), 0, self.learn_eps)
            )
            self.bond_layers.append(
                BondEncoder(hidden_dim)
            )

        self.linears_prediction = nn.Linear(hidden_dim, output_dim)

        self.drop = nn.Dropout(final_dropout)

        if graph_pooling_type == 'sum':
            self.pool = SumPooling()
        elif graph_pooling_type == 'mean':
            self.pool = AvgPooling()
        elif graph_pooling_type == 'max':
            self.pool = MaxPooling()
        else:
            raise NotImplementedError

    def forward(self, g, h_node, h_edge):
        h_n = self.atom_encoder(h_node)
        split_list = g.batch_num_nodes

        for i in range(self.num_layers - 1):
            x = h_n
            h_e = self.bond_layers[i](h_edge)
            h_n = self.ginlayers[i](g, split_list, h_n, h_e)

            if i != self.num_layers - 2:
                h_n = F.relu(h_n)

            h_n += x

        score_over_layer = 0
        pooled_h = self.pool(g, h_n)
        score_over_layer += self.drop(self.linears_prediction(pooled_h))

        return score_over_layer