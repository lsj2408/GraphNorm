import argparse, random, os, glob
import numpy as np
import torch
import torch.nn as nn

from dgl_model.gin_all import GIN
from dgl_model.gcn_all import GCN, GCN_dp

from torch.utils.data import DataLoader

from ogb.graphproppred.dataset_dgl import DglGraphPropPredDataset, collate_dgl
from ogb.graphproppred import Evaluator

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    if args.gpu >= 0:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def load_model(args, model):
    path = glob.glob(args.model_path+'*.pth')
    assert len(path) == 1

    if not os.path.isfile(path[0]):
        raise('Not find model')
    else:
        params = torch.load(path[0])
        model.load_state_dict(params)
    return model

def evaluate(model, dataloader, loss_fcn, evaluator):
    model.eval()

    total = 0
    total_loss = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for data in dataloader:
            graphs, labels = data
            node_feat = graphs.ndata['feat'].cuda()
            edge_feat = graphs.edata['feat'].cuda()
            labels = labels.cuda()
            total += len(labels)
            outputs = model(graphs, node_feat, edge_feat)

            y_true.append(labels.view(outputs.shape).detach().cpu())
            y_pred.append(outputs.detach().cpu())

            is_valid = labels == labels
            loss = loss_fcn(outputs.to(torch.float32)[is_valid], labels.to(torch.float32)[is_valid])
            total_loss += loss * len(labels)

    loss = 1.0 * total_loss / total

    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()

    input_dict = {'y_true': y_true, 'y_pred': y_pred}
    return loss, evaluator.eval(input_dict)

def task_data(args, dataset=None):

    # DATA_ROOT = '/mnt/localdata/users/shengjie/ogb_ws/data/dataset'
    # step 0: setting for gpu

    if args.gpu >= 0:
        torch.cuda.set_device(args.gpu)

    # step 1: prepare dataset

    if dataset is None:
        dataset = DglGraphPropPredDataset(name=args.dataset, root=args.data_dir)
    splitted_idx = dataset.get_idx_split()

    # step 2: prepare data_loader

    train_loader = DataLoader(dataset[splitted_idx['train']], batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_dgl, num_workers=4)
    valid_loader = DataLoader(dataset[splitted_idx['valid']], batch_size=args.batch_size, shuffle=False,
                              collate_fn=collate_dgl, num_workers=4)
    test_loader = DataLoader(dataset[splitted_idx['test']], batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate_dgl, num_workers=4)

    evaluator = Evaluator(args.dataset)

    return dataset, evaluator, train_loader, valid_loader, test_loader

def task_model(args, dataset):

    #  step 1: prepare model
    assert args.model in ['GIN', 'GCN', 'GCN_dp']
    if args.model == 'GIN':
        model = GIN(
            args.n_layers, args.n_mlp_layers,
            args.n_hidden,
            dataset.num_tasks,
            args.dropout, args.learn_eps,
            args.graph_pooling_type, args.neighbor_pooling_type,
            args.norm_type
        )
    elif args.model == 'GCN':
        model = GCN(
            args.n_layers,
            args.n_hidden,
            dataset.num_tasks,
            args.dropout, args.learn_eps,
            args.graph_pooling_type,
            args.norm_type
        )
    elif args.model == 'GCN_dp':
        model = GCN_dp(
            args.n_layers,
            args.n_hidden,
            dataset.num_tasks,
            args.dropout, args.learn_eps,
            args.graph_pooling_type,
            args.norm_type
        )
    else:
        raise('Not supporting such model!')

    saved_model = torch.load(args.model_path)

    model.load_state_dict(saved_model)

    if args.gpu >= 0:
        model = model.cuda()

    # step 2: prepare loss

    loss_fcn = nn.BCEWithLogitsLoss()

    # step 3: prepare optimizer

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    return model, loss_fcn, optimizer

def train(train_loader, valid_loader, test_loader, model, loss_fcn, evaluator):


    test_loss, test_acc = evaluate(model, test_loader, loss_fcn, evaluator)
    valid_loss, valid_acc = evaluate(model, valid_loader, loss_fcn, evaluator)
    train_loss, train_acc = evaluate(model, train_loader, loss_fcn, evaluator)

    return float(list(train_acc.items())[0][1]), float(list(valid_acc.items())[0][1]), float(list(test_acc.items())[0][1])

def main(args):
    dataset, evaluator, train_loader, valid_loader, test_loader = task_data(args)

    # for seed in range(args.seed):
    seed = args.seed
    set_seed(seed)
    train_metric_list = []
    valid_metric_list = []
    test_metric_list = []
    model_path_list = glob.glob(args.model_path + '/*.pth')
    for paths in model_path_list:
        args.model_path = paths
        model, loss_fcn, optimizer = task_model(args, dataset)
        train_acc, valid_acc, test_acc = train(train_loader, valid_loader, test_loader,
                                            model, loss_fcn, evaluator)
        train_metric_list.append(train_acc)
        valid_metric_list.append(valid_acc)
        test_metric_list.append(test_acc)

    print('*************************** RESULT ***************************')
    print('Train metric: Avg {:.4f} | Max {:.4f} | Min {:.4f} | Std {:.4f}'.format(
        np.mean(train_metric_list), np.max(train_metric_list), np.min(train_metric_list), np.std(train_metric_list)
    ))
    print('Valid metric: Avg {:.4f} | Max {:.4f} | Min {:.4f} | Std {:.4f}'.format(
        np.mean(valid_metric_list), np.max(valid_metric_list), np.min(valid_metric_list), np.std(valid_metric_list)
    ))
    print('Test metric: Avg {:.4f} | Max {:.4f} | Min {:.4f} | Std {:.4f}'.format(
        np.mean(test_metric_list), np.max(test_metric_list), np.min(test_metric_list), np.std(test_metric_list)
    ))
    print('***************************  END  ****************************')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OGB-molecular')

    # 1) general params
    parser.add_argument("--gpu", type=int, default=-1,
                        help="gpu")
    parser.add_argument("--seed", type=int, default=1,
                        help='random seed')
    parser.add_argument("--log_dir", type=str, help='path to the output log file')

    parser.add_argument("--model_path", type=str, help='path to save the model')

    parser.add_argument("--data_dir", type=str, help='path to data')

    parser.add_argument('--exp', type=str, help='experiment name')
    parser.add_argument(
        '--dataset', type=str, default='ogbg-molhiv',
        # choices=['ogbg-mol-hiv','ogbg-mol-pcba', 'ogbg-mol-muv'],
        help='name of dataset (default: ogbg-mol-hiv)'
    )

    # 2) model params
    parser.add_argument("--model", type=str, default='GIN',
                        help='graph models')
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="learning rate")
    parser.add_argument("--dropout", type=float, default=0.5,
                        help='dropout probability')
    parser.add_argument("--epoch", type=int, default=100,
                        help="number of training epochs")
    parser.add_argument("--n_hidden", type=int, default=300,
                        help='number of hidden gcn layers')
    parser.add_argument("--weight_decay", type=float, default=0.0,
                        help='Weight for L2 Loss')
    parser.add_argument("--n_layers", type=int, default=5,
                        help='num of layers')
    parser.add_argument("--n_mlp_layers", type=int, default=2,
                        help='num of mlp layers')

    parser.add_argument('--batch_size', type=int, default=128,
        help='batch size for training and validation (default: 32)'
    )

    parser.add_argument('--graph_pooling_type', type=str,
        default="mean", choices=["sum", "mean", "max"],
        help='type of graph pooling: sum, mean or max')
    parser.add_argument('--neighbor_pooling_type', type=str,
        default="sum", choices=["sum", "mean", "max"],
        help='type of neighboring pooling: sum, mean or max')
    parser.add_argument('--learn_eps', action="store_true",
        help='learn the epsilon weighting')
    # 3) specific params

    parser.add_argument('--norm_type', type=str,
                        default='gn', choices=['bn', 'gn'],
                        help='type of normalization')

    parser.add_argument('--patience', type=int, default=10,
                        help='patience of early-stopping')


    args = parser.parse_args()

    # args.n_layers = 5
    # args.n_mlp_layers = 2
    #
    # args.neighbor_pooling_type = 'sum'
    # args.model = 'GIN'
    #
    # args.norm_type = 'gn'
    # args.gpu = 3
    #
    # args.data_dir = '/mnt/localdata/users/shengjie/ogbg_ws/data/dataset/'
    # # args.model_path = '/mnt/localdata/users/shengjie/ogbg_ws/model/Pre-train-model/GCN-GN/'
    # args.model_path = '/mnt/localdata/users/shengjie/ogbg_ws/model/Pre-train-model/GIN-GN/'
    #
    # #
    # args.dataset = 'ogbg-molhiv'
    # # [0, 0.5]
    # args.dropout = 0.5
    # # [16, 32]
    # args.n_hidden = 300
    # # [32, 128]
    # #
    # args.batch_size = 128
    # # ['sum', 'mean']
    # #
    # args.graph_pooling_type = 'mean'
    # *********************************************************

    main(args)





