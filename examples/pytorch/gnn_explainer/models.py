import torch as th
import torch.nn as nn
import torch.nn.functional as F

import torch
import math
import numpy as np
from torch.nn.parameter import Parameter
import dgl
import dgl.function as fn
from dgl.nn import GATConv

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, residual=False, variant=False):
        super(GraphConvolution, self).__init__()
        self.variant = variant
        if self.variant:
            self.in_features = 2*in_features
        else:
            self.in_features = in_features

        self.out_features = out_features
        self.residual = residual
        self.weight = Parameter(torch.FloatTensor(self.in_features,self.out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, graph, n_features , h0 , lamda, alpha, l, e_weights = None):
        theta = math.log(lamda/l+1)
        graph.ndata['h'] = n_features
        if e_weights == None:
            graph.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'h'))
        else:
            graph.edata['ew'] = e_weights
            graph.update_all(fn.u_mul_e('h', 'ew', 'm'), fn.mean('m', 'h'))
        hi = graph.ndata['h']
        if self.variant:
            support = torch.cat([hi,h0],1)
            r = (1-alpha)*hi+alpha*h0
        else:
            support = (1-alpha)*hi+alpha*h0
            r = support
        output = theta*torch.mm(support, self.weight)+(1-theta)*r
        graph.ndata['h'] = output
        return output

class GCNII(nn.Module):
    def __init__(self, nfeat, nlayers,nhidden, nclass, dropout, lamda, alpha, variant):
        super(GCNII, self).__init__()
        self.convs = nn.ModuleList()
        for _ in range(nlayers):
            self.convs.append(GraphConvolution(nhidden, nhidden,variant=variant))
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(nfeat, nhidden))
        self.fcs.append(nn.Linear(nhidden, nclass))
        self.params1 = list(self.convs.parameters())
        self.params2 = list(self.fcs.parameters())
        self.act_fn = nn.ReLU()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda

    def forward(self,graph, x,  e_weights = None):
        _layers = []
        x = F.dropout(x, self.dropout, training=self.training)
        layer_inner = self.act_fn(self.fcs[0](x))
        _layers.append(layer_inner)
        for i,con in enumerate(self.convs):
            layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            layer_inner = self.act_fn(con(graph, layer_inner, _layers[0],self.lamda,self.alpha,i+1, e_weights))
        layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
        layer_inner = self.fcs[-1](layer_inner)
        return layer_inner



class dummy_layer(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(dummy_layer, self).__init__()
        self.layer = nn.Linear(in_dim * 2, out_dim, bias=True)

    def forward(self, graph, n_feats, e_weights=None):
        graph.ndata['h'] = n_feats

        if e_weights == None:
            graph.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'h'))
        else:
            graph.edata['ew'] = e_weights
            graph.update_all(fn.u_mul_e('h', 'ew', 'm'), fn.mean('m', 'h'))

        graph.ndata['h'] = self.layer(th.cat([graph.ndata['h'], n_feats], dim=-1))

        output = graph.ndata['h']
        return output


class dummy_gnn_model(nn.Module):

    """
    A dummy gnn model, which is same as graph sage, but could adopt edge mask in forward

    """
    def __init__(self,
                 in_dim,
                 hid_dim,
                 out_dim):
        super(dummy_gnn_model, self).__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        
        self.in_layer = dummy_layer(self.in_dim, self.hid_dim)
        self.hid_layer = dummy_layer(self.hid_dim, self.hid_dim)
        self.out_layer = dummy_layer(self.hid_dim, self.out_dim)

    def forward(self, graph, n_feat, edge_weights=None):

        h = self.in_layer(graph, n_feat, edge_weights)
        h = F.relu(h)
        h = self.hid_layer(graph, h, edge_weights)
        h = F.relu(h)
        h = self.out_layer(graph, h, edge_weights)

        return h

class GAT(nn.Module):
    def __init__(self,
                 g,
                 num_layers,
                 in_dim,
                 num_hidden,
                 num_classes,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual):
        super(GAT, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        # input projection (no residual)
        self.gat_layers.append(GATConv(
            in_dim, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATConv(
                num_hidden * heads[l-1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation))
        # output projection
        self.gat_layers.append(GATConv(
            num_hidden * heads[-2], num_classes, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, None))

    def forward(self, inputs):
        h = inputs
        for l in range(self.num_layers):
            h = self.gat_layers[l](self.g, h).flatten(1)
        # output projection
        logits = self.gat_layers[-1](self.g, h).mean(1)
        return logits