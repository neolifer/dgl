# The training codes of the dummy model


import os
import argparse
import dgl
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl import save_graphs
from models import dummy_gnn_model, GCNII, GAT
from gengraph import gen_syn1, gen_syn2, gen_syn3, gen_syn4, gen_syn5
import numpy as np
from tqdm.notebook import tqdm
import random
import sys


def main(args):
    # load dataset
    if args.dataset == 'syn1':
        g, labels, name = gen_syn1()
    elif args.dataset == 'syn2':
        g, labels, name = gen_syn2()
    elif args.dataset == 'syn3':
        g, labels, name = gen_syn3()
    elif args.dataset == 'syn4':
        g, labels, name = gen_syn4()
    elif args.dataset == 'syn5':
        g, labels, name = gen_syn5()
    else:
        raise NotImplementedError
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    #Transform to dgl graph. 
    graph = dgl.from_networkx(g) 
    labels = th.tensor(labels, dtype=th.long)
    graph.ndata['label'] = labels
    graph.ndata['feat'] = th.randn(graph.number_of_nodes(), args.feat_dim)
    hid_dim = th.tensor(args.hidden_dim, dtype=th.long)
    label_dict = {'hid_dim':hid_dim}

    # save graph for later use
    save_graphs(filename='./'+args.dataset+'.bin', g_list=[graph], labels=label_dict)

    num_classes = max(graph.ndata['label']).item() + 1
    n_feats = graph.ndata['feat']

    #create model
    dummy_model =      model = GAT(graph,
                                   args.num_layers,
                                   args.feat_dim,
                                   args.num_hidden,
                                   num_classes,
                                   ([args.num_heads] * args.num_layers) + [args.num_out_heads],
                                   F.elu,
                                   args.in_drop,
                                   args.attn_drop,
                                   args.negative_slope,
                                   args.residual)
    loss_fn = nn.CrossEntropyLoss()
    optim = th.optim.Adam(dummy_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


    # train and output
    for epoch in tqdm(range(args.epochs)):

        dummy_model.train()

        logits = dummy_model(graph, n_feats)

        loss = loss_fn(logits, labels)
        acc = th.sum(logits.argmax(dim=1) == labels).item() / len(labels)
        
        optim.zero_grad()
        loss.backward()
        optim.step()

        print('In Epoch: {:03d}; Acc: {:.4f}; Loss: {:.6f}'.format(epoch, acc, loss.item()))

    # save model
    model_stat_dict = dummy_model.state_dict()
    model_path = os.path.join('./', 'dummy_model_{}.pth'.format(args.dataset))
    th.save(model_stat_dict, model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dummy model training')
    parser.add_argument('--dataset', type=str, default='syn1', help='The dataset used for training the model.')
    parser.add_argument('--feat_dim', type=int, default=10, help='The feature dimension.')
    parser.add_argument('--hidden_dim', type=int, default=40, help='The hidden dimension.')
    #hyper parameters for GCNII
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    # parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')
    # parser.add_argument('--lr', type=float, default=0.005, help='learning rate.')
    # parser.add_argument('--wd1', type=float, default=0.0, help='weight decay (L2 loss on parameters).')
    # parser.add_argument('--wd2', type=float, default=0, help='weight decay (L2 loss on parameters).')
    # parser.add_argument('--layer', type=int, default=6, help='Number of layers.')
    # parser.add_argument('--hidden', type=int, default=128, help='hidden dimensions.')
    # parser.add_argument('--dropout', type=float, default=0, help='Dropout rate (1 - keep probability).')
    # parser.add_argument('--patience', type=int, default=100, help='Patience')
    # parser.add_argument('--data', default='cora', help='dateset')
    # parser.add_argument('--dev', type=int, default=0, help='device id')
    # parser.add_argument('--alpha', type=float, default=0.1, help='alpha_l')
    # parser.add_argument('--lamda', type=float, default=0.5, help='lamda.')
    # parser.add_argument('--variant', action='store_true', default=False, help='GCN* model.')
    # parser.add_argument('--test', action='store_true', default=False, help='evaluation on test set.')
    #hyper parameters for GAT
    parser.add_argument("--gpu", type=int, default=-1,
                        help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument("--epochs", type=int, default=1000,
                        help="number of training epochs")
    parser.add_argument("--num-heads", type=int, default=8,
                        help="number of hidden attention heads")
    parser.add_argument("--num-out-heads", type=int, default=1,
                        help="number of output attention heads")
    parser.add_argument("--num-layers", type=int, default=2,
                        help="number of hidden layers")
    parser.add_argument("--num-hidden", type=int, default=16,
                        help="number of hidden units")
    parser.add_argument("--residual", action="store_true", default=False,
                        help="use residual connection")
    parser.add_argument("--in-drop", type=float, default=0,
                        help="input feature dropout")
    parser.add_argument("--attn-drop", type=float, default=0,
                        help="attention dropout")
    parser.add_argument("--lr", type=float, default=0.005,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=0,
                        help="weight decay")
    parser.add_argument('--negative-slope', type=float, default=0.2,
                        help="the negative slope of leaky relu")
    parser.add_argument('--early-stop', action='store_true', default=False,
                        help="indicates whether to use early stop or not")
    parser.add_argument('--fastmode', action="store_true", default=False,
                        help="skip re-evaluate the validation set")
    args = parser.parse_args()
    print(args)

    main(args)
    
