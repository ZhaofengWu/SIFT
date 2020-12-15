import torch.nn.functional as F
import torch.nn as nn

from dgl.nn.pytorch import RelGraphConv


class GNN(nn.Module):
    def __init__(self, h_dim, num_relations, num_hidden_layers=1, dropout=0, activation=F.relu):
        super().__init__()
        self.h_dim = h_dim
        self.num_relations = num_relations
        self.dropout = dropout
        self.activation = activation
        self.layers = nn.ModuleList([self.create_graph_layer() for _ in range(num_hidden_layers)])

    def create_graph_layer(self):
        raise NotImplementedError

    def forward(self, graph, initial_embeddings):
        if len(initial_embeddings) != len(graph.nodes):
            raise ValueError('Node embedding initialization shape mismatch')
        h = initial_embeddings
        for layer in self.layers:
            h = self.forward_graph_layer(layer, graph, h)
        return h

    def forward_graph_layer(self, layer, graph, h):
        raise NotImplementedError


class RGCN(GNN):
    def __init__(self, num_bases=-1, **kwargs):
        self.num_bases = None if num_bases < 0 else num_bases
        super().__init__(**kwargs)

    def create_graph_layer(self):
        return RelGraphConv(
            self.h_dim,
            self.h_dim,
            self.num_relations,
            "basis",
            self.num_bases,
            activation=self.activation,
            self_loop=True,
            dropout=self.dropout,
        )

    def forward_graph_layer(self, layer, graph, h):
        return layer(
            graph,
            h,
            graph.edata['type'] if 'type' in graph.edata else h.new_empty(0),
            graph.edata['norm'] if 'norm' in graph.edata else h.new_empty(0),
        )
