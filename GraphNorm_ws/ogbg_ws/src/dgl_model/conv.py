import torch
import torch.nn as nn
from dgl import function as fn

class GINConv(nn.Module):
    def __init__(self, apply_func, aggregator_type, init_eps=0, learn_eps=False):
        super(GINConv, self).__init__()
        self.apply_func = apply_func
        if aggregator_type == 'sum':
            self._reducer = fn.sum
        elif aggregator_type == 'max':
            self._reducer = fn.max
        elif aggregator_type == 'mean':
            self._reducer = fn.mean
        else:
            raise KeyError('Aggregator type {} not recognized.'.format(aggregator_type))

        if learn_eps:
            self.eps = torch.nn.Parameter(torch.FloatTensor([init_eps]))
        else:
            self.register_buffer('eps', torch.FloatTensor([init_eps]))

    def forward(self, graph, feat):
        graph = graph.local_var()
        graph.ndata['h'] = feat
        graph.update_all(fn.copy_u('h', 'm'),
                         self._reducer('m', 'neigh'))
        rst = (1 + self.eps) * feat + graph.ndata['neigh']
        if self.apply_func is not None:
            rst = self.apply_func(rst)
        return rst