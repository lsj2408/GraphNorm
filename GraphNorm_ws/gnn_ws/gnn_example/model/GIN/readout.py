import torch as th
import torch.nn as nn
import numpy as np

from dgl.batched_graph import sum_nodes, mean_nodes, max_nodes


class SumPooling(nn.Module):
    r"""Apply sum pooling over the nodes in the graph.

    .. math::
        r^{(i)} = \sum_{k=1}^{N_i} x^{(i)}_k
    """
    def __init__(self):
        super(SumPooling, self).__init__()

    def forward(self, graph, feat):
        r"""Compute sum pooling.


        Parameters
        ----------
        graph : DGLGraph or BatchedDGLGraph
            The graph.
        feat : torch.Tensor
            The input feature with shape :math:`(N, *)` where
            :math:`N` is the number of nodes in the graph.

        Returns
        -------
        torch.Tensor
            The output feature with shape :math:`(*)` (if
            input graph is a BatchedDGLGraph, the result shape
            would be :math:`(B, *)`.
        """
        with graph.local_scope():
            graph.ndata['h'] = feat
            readout = sum_nodes(graph, 'h')
            return readout


class AvgPooling(nn.Module):
    r"""Apply average pooling over the nodes in the graph.

    .. math::
        r^{(i)} = \frac{1}{N_i}\sum_{k=1}^{N_i} x^{(i)}_k
    """
    def __init__(self):
        super(AvgPooling, self).__init__()

    def forward(self, graph, feat):
        r"""Compute average pooling.

        Parameters
        ----------
        graph : DGLGraph or BatchedDGLGraph
            The graph.
        feat : torch.Tensor
            The input feature with shape :math:`(N, *)` where
            :math:`N` is the number of nodes in the graph.

        Returns
        -------
        torch.Tensor
            The output feature with shape :math:`(*)` (if
            input graph is a BatchedDGLGraph, the result shape
            would be :math:`(B, *)`.
        """
        with graph.local_scope():
            graph.ndata['h'] = feat
            readout = mean_nodes(graph, 'h')
            return readout


class MaxPooling(nn.Module):
    r"""Apply max pooling over the nodes in the graph.

    .. math::
        r^{(i)} = \max_{k=1}^{N_i}\left( x^{(i)}_k \right)
    """
    def __init__(self):
        super(MaxPooling, self).__init__()

    def forward(self, graph, feat):
        r"""Compute max pooling.

        Parameters
        ----------
        graph : DGLGraph or BatchedDGLGraph
            The graph.
        feat : torch.Tensor
            The input feature with shape :math:`(N, *)` where
            :math:`N` is the number of nodes in the graph.

        Returns
        -------
        torch.Tensor
            The output feature with shape :math:`(*)` (if
            input graph is a BatchedDGLGraph, the result shape
            would be :math:`(B, *)`.
        """
        with graph.local_scope():
            graph.ndata['h'] = feat
            readout = max_nodes(graph, 'h')
            return readout