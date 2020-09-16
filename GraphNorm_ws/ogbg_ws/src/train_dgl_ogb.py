import argparse, time, random, os
import numpy as np
import torch
import torch.nn as nn

from dgl_model.gin_all import GIN
from dgl_model.gcn_all import GCN

from torch.utils.data import DataLoader

from ogb.graphproppred.dataset_dgl import DglGraphPropPredDataset, collate_dgl
from ogb.graphproppred import Evaluator

from utils.scheduler import LinearSchedule
from utils.earlystopper import EarlyStopping

def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    if args.gpu >= 0:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def grad_norm(net):
    ret = 0
    for param in net.parameters():
        ret += torch.norm(param.grad.data)**2 if param.grad is not None else 0.0
    return torch.sqrt(ret).data.cpu().numpy()

def param_norm(net):
    ret = 0
    for param in net.parameters():
        ret += torch.norm(param.data)**2
    return torch.sqrt(ret).data.cpu().numpy()

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
    assert args.model in ['GIN', 'GCN']
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
    else:
        raise('Not supporting such model!')
    if args.gpu >= 0:
        model = model.cuda()

    # step 2: prepare loss
    loss_fcn = nn.BCEWithLogitsLoss()

    # step 3: prepare optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    return model, loss_fcn, optimizer

def train(args, train_loader, valid_loader, test_loader, model, loss_fcn, optimizer, evaluator):

    scheduler = LinearSchedule(optimizer, args.epoch)

    EarlyStopper = EarlyStopping(args.patience, path=args.model_path)

    dur = []
    record = {}
    grad_record = {}
    param_record = {}
    final_acc = 0.0
    for epoch in range(args.epoch):
        model.train()
        t0 = time.time()

        for graphs, labels in train_loader:
            labels = labels.cuda()
            node_features = graphs.ndata['feat'].cuda()
            edge_features = graphs.edata['feat'].cuda()
            outputs = model(graphs, node_features, edge_features)

            optimizer.zero_grad()

            is_valid = labels == labels
            loss = loss_fcn(outputs.to(torch.float32)[is_valid], labels.to(torch.float32)[is_valid])

            loss.backward()
            optimizer.step()

        test_loss, test_acc = evaluate(model, test_loader, loss_fcn, evaluator)
        valid_loss, valid_acc = evaluate(model, valid_loader, loss_fcn, evaluator)
        train_loss, train_acc = evaluate(model, train_loader, loss_fcn, evaluator)

        dur.append(time.time() - t0)
        print('Average Epoch Time {:.4f}'.format(float(sum(dur) / len(dur))))
        print('Train metric {:.4f}'.format(float(list(train_acc.items())[0][1])))
        print('Valid metric {:.4f}'.format(float(list(valid_acc.items())[0][1])))
        print('Test metric {:.4f}'.format(float(list(test_acc.items())[0][1])))

        record[epoch] = (np.mean(dur), train_loss.item(), float(list(train_acc.items())[0][1]),
                         valid_loss.item(), float(list(valid_acc.items())[0][1]),
                         test_loss.item(), float(list(test_acc.items())[0][1]))

        if args.log_norm:
            grad_n = grad_norm(model)
            param_n = param_norm(model)
            grad_record[epoch] = grad_n
            param_record[epoch] = param_n

        if final_acc < float(list(valid_acc.items())[0][1]):
            final_acc = float(list(valid_acc.items())[0][1])
            EarlyStopper.save_checkpoint(model)

        scheduler.step()

    print("Test metric {:.4f}".format(final_acc))
    return record, grad_record, param_record

def main(args):
    dataset, evaluator, train_loader, valid_loader, test_loader = task_data(args)
    result_record = {}
    grad_record = {}
    param_record = {}
    # for seed in range(args.seed):
    seed = args.seed
    set_seed(seed)
    model, loss_fcn, optimizer = task_model(args, dataset)
    result_record[seed], grad_record[seed], param_record[seed] = train(args, train_loader, valid_loader, test_loader,
                                            model, loss_fcn, optimizer, evaluator)
    return result_record, grad_record, param_record

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OGB-molecular')

    # 1) general params
    parser.add_argument("--gpu", type=int, default=0,
                        help="gpu")
    parser.add_argument("--seed", type=int, default=1,
                        help='random seed')
    parser.add_argument("--log_dir", type=str, help='path to the output log file')

    parser.add_argument("--data_dir", type=str, help='path to the data')

    parser.add_argument("--model_path", type=str, help='path to save the model')

    parser.add_argument('--exp', type=str, help='experiment name')
    parser.add_argument(
        '--dataset', type=str, default='ogbg-molhiv',
        help='name of dataset (default: ogbg-molhiv)'
    )

    # 2) model params
    parser.add_argument("--model", type=str, default='GIN',
                        help='graph models')
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="learning rate")
    parser.add_argument("--dropout", type=float, default=0.5,
                        help='dropout probability')
    parser.add_argument("--epoch", type=int, default=25,
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

    parser.add_argument('--log_norm', action='store_true',
                        help='log normalization information')

    parser.add_argument('--norm_type', type=str,
                        default='gn',
                        help='type of normalization')

    parser.add_argument('--patience', type=int, default=10,
                        help='patience of early-stopping')

    args = parser.parse_args()

    args.model_path = os.path.join(args.model_path,
                             'seed-%d-%s-bs-%d-dp-%.2f-hidden-%d-wd-%.4f-nl-%d-epoch-%d-lr-%.4f-Norm-%s-log.pth' %
                             (args.seed, args.exp, args.batch_size, args.dropout, args.n_hidden, args.weight_decay,
                              args.n_layers, args.epoch, args.lr, args.norm_type))


    print(args)
    result, grad, param = main(args)

    def output_result(args, result, grad=None, param=None):
        raw_record = {}
        for seed in result.keys():
            content = result[seed]
            for epoch in content.keys():
                time, train_loss, train_acc, \
                valid_loss, valid_acc, test_loss, test_acc = content[epoch]
                if epoch not in raw_record.keys():
                    raw_record[epoch] = {
                        'time': [],
                        'train_loss':[],
                        'train_acc':[],
                        'test_loss':[],
                        'test_acc':[],
                        'valid_loss':[],
                        'valid_acc':[]
                    }

                raw_record[epoch]['time'].append(time)
                raw_record[epoch]['train_loss'].append(train_loss)
                raw_record[epoch]['train_acc'].append(train_acc)
                raw_record[epoch]['valid_loss'].append(valid_loss)
                raw_record[epoch]['valid_acc'].append(valid_acc)
                raw_record[epoch]['test_loss'].append(test_loss)
                raw_record[epoch]['test_acc'].append(test_acc)


        assert args.log_norm and grad is not None and param is not None

        grad_record = {}
        if grad is not None:
            for seed in grad.keys():
                content = grad[seed]
                for epoch in content.keys():
                    grad_n = content[epoch]
                    if epoch not in grad_record.keys():
                        grad_record[epoch] = {
                            'grad': [],
                        }

                    grad_record[epoch]['grad'].append(grad_n)

        param_record = {}
        if param is not None:
            for seed in param.keys():
                content = param[seed]
                for epoch in content.keys():
                    param_n = content[epoch]
                    if epoch not in param_record.keys():
                        param_record[epoch] = {
                            'param': [],
                        }

                    param_record[epoch]['param'].append(param_n)

        import xlwt
        import os
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)
        save_path = os.path.join(args.log_dir, '%s-bs-%d-dp-%.2f-hidden-%d-wd-%.4f-nl-%d-epoch-%d-lr-%.4f-Norm-%s-log.xls' %
                                 (args.exp, args.batch_size, args.dropout, args.n_hidden, args.weight_decay,
                                  args.n_layers, args.epoch, args.lr, args.norm_type))
        f = xlwt.Workbook()
        sheet = f.add_sheet("result")
        sheet.write(0, 0, 'time')
        sheet.write(0, 1, 'train_loss')
        sheet.write(0, 2, 'train_acc')
        sheet.write(0, 3, 'test_loss')
        sheet.write(0, 4, 'test_acc')
        sheet.write(0, 5, 'test_max_acc')
        sheet.write(0, 6, 'test_min_acc')
        sheet.write(0, 7, 'test_std_acc')
        sheet.write(0, 8, 'train_max_acc')
        sheet.write(0, 9, 'train_min_acc')
        sheet.write(0, 10, 'train_std_acc')
        sheet.write(0, 11, 'valid_loss')
        sheet.write(0, 12, 'valid_acc')
        sheet.write(0, 13, 'valid_max_acc')
        sheet.write(0, 14, 'valid_min_acc')
        sheet.write(0, 15, 'valid_std_acc')


        for epoch in range(len(raw_record.keys())):
            time = np.mean(raw_record[epoch]['time'])
            train_loss = np.mean(raw_record[epoch]['train_loss'])
            train_acc = np.mean(raw_record[epoch]['train_acc'])

            test_loss = np.mean(raw_record[epoch]['test_loss'])
            test_acc = np.mean(raw_record[epoch]['test_acc'])

            valid_loss = np.mean(raw_record[epoch]['valid_loss'])
            valid_acc = np.mean(raw_record[epoch]['valid_acc'])

            train_max_acc = np.max(raw_record[epoch]['train_acc'])
            train_min_acc = np.min(raw_record[epoch]['train_acc'])
            train_std_acc = np.std(raw_record[epoch]['train_acc'])
            test_max_acc = np.max(raw_record[epoch]['test_acc'])
            test_min_acc = np.min(raw_record[epoch]['test_acc'])
            test_std_acc = np.std(raw_record[epoch]['test_acc'])
            valid_max_acc = np.max(raw_record[epoch]['valid_acc'])
            valid_min_acc = np.min(raw_record[epoch]['valid_acc'])
            valid_std_acc = np.std(raw_record[epoch]['valid_acc'])
            sheet.write(epoch+1, 0, time)
            sheet.write(epoch+1, 1, train_loss)
            sheet.write(epoch+1, 2, train_acc)
            sheet.write(epoch+1, 3, test_loss)
            sheet.write(epoch+1, 4, test_acc)
            sheet.write(epoch+1, 5, test_max_acc)
            sheet.write(epoch+1, 6, test_min_acc)
            sheet.write(epoch+1, 7, test_std_acc)
            sheet.write(epoch+1, 8, train_max_acc)
            sheet.write(epoch+1, 9, train_min_acc)
            sheet.write(epoch+1, 10, train_std_acc)
            sheet.write(epoch + 1, 11, valid_loss)
            sheet.write(epoch + 1, 12, valid_acc)
            sheet.write(epoch + 1, 13, valid_max_acc)
            sheet.write(epoch + 1, 14, valid_min_acc)
            sheet.write(epoch + 1, 15, valid_std_acc)


        if args.log_norm:
            sheet = f.add_sheet("grad_param")
            sheet.write(0, 0, 'grad_norm')
            sheet.write(0, 1, 'grad_norm_max')
            sheet.write(0, 2, 'grad_norm_min')
            sheet.write(0, 3, 'grad_norm_std')

            sheet.write(0, 5, 'param_norm')
            sheet.write(0, 6, 'param_norm_max')
            sheet.write(0, 7, 'param_norm_min')
            sheet.write(0, 8, 'param_norm_std')

            for epoch in range(len(grad_record.keys())):
                grad_n = np.mean(grad_record[epoch]['grad'])
                param_n = np.mean(param_record[epoch]['param'])
                grad_n_max = np.max(grad_record[epoch]['grad'])
                param_n_max = np.max(param_record[epoch]['param'])
                grad_n_min = np.min(grad_record[epoch]['grad'])
                param_n_min = np.min(param_record[epoch]['param'])
                grad_n_std = np.std(grad_record[epoch]['grad'])
                param_n_std = np.std(param_record[epoch]['param'])

                sheet.write(epoch + 1, 0, float(grad_n))
                sheet.write(epoch + 1, 1, float(grad_n_max))
                sheet.write(epoch + 1, 2, float(grad_n_min))
                sheet.write(epoch + 1, 3, float(grad_n_std))
                sheet.write(epoch + 1, 5, float(param_n))
                sheet.write(epoch + 1, 6, float(param_n_max))
                sheet.write(epoch + 1, 7, float(param_n_min))
                sheet.write(epoch + 1, 8, float(param_n_std))

        f.save(save_path)

    output_result(args, result, grad, param)



