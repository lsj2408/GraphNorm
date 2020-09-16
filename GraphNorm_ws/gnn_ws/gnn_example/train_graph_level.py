import argparse, time, random, os
import numpy as np
import torch
import torch.nn as nn

from model.GIN.gin_all_fast import GIN
from model.GCN.gcn_all import GCN
from Temp.dataset import GINDataset
from utils.GIN.data_loader import GraphDataLoader, collate
from utils.scheduler import LinearSchedule

def set_seed(seed):
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

def evaluate(model, dataloader, loss_fcn):
    model.eval()

    total = 0
    total_loss = 0
    total_correct = 0
    with torch.no_grad():
        for data in dataloader:
            graphs, labels = data
            feat = graphs.ndata['attr'].cuda()
            labels = labels.cuda()
            total += len(labels)
            outputs = model(graphs, feat)
            _, predicted = torch.max(outputs.data, 1)

            total_correct += (predicted == labels.data).sum().item()
            loss = loss_fcn(outputs, labels)

            total_loss += loss * len(labels)

    loss, acc = 1.0 * total_loss / total, 1.0 * total_correct / total
    return loss, acc

def task_data(args, dataset=None):

    # step 0: setting for gpu
    if args.gpu >= 0:
        torch.cuda.set_device(args.gpu)
    # step 1: prepare dataset
    if dataset is None:
        dataset = GINDataset(args.dataset, args.self_loop, args.degree_as_label)

    # step 2: prepare data_loader
    train_loader, valid_loader = GraphDataLoader(
        dataset, batch_size=args.batch_size, device=args.gpu,
        collate_fn=collate, seed=args.seed, shuffle=True,
        split_name=args.split_name, fold_idx=args.fold_idx
    ).train_valid_loader()


    return dataset, train_loader, valid_loader

def task_model(args, dataset):

    #  step 1: prepare model
    assert args.model in ['GIN', 'GCN']
    if args.model == 'GIN':
        model = GIN(
            args.n_layers, args.n_mlp_layers,
            dataset.dim_nfeats, args.n_hidden, dataset.gclasses,
            args.dropout, args.learn_eps,
            args.graph_pooling_type, args.neighbor_pooling_type,
            args.norm_type
        )
    elif args.model == 'GCN':
        model = GCN(
            args.n_layers, dataset.dim_nfeats, args.n_hidden,
            dataset.gclasses, args.dropout, args.graph_pooling_type,
            norm_type=args.norm_type
        )
    else:
        raise('Not supporting such model!')
    if args.gpu >= 0:
        model = model.cuda()

    # step 2: prepare loss
    loss_fcn = nn.CrossEntropyLoss()

    # step 3: prepare optimizer

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    return model, loss_fcn, optimizer

def train(args, train_loader, valid_loader, model, loss_fcn, optimizer):

    scheduler = LinearSchedule(optimizer, args.epoch)

    dur = []
    record = {}
    grad_record = {}
    param_record = {}

    for epoch in range(args.epoch):
        model.train()
        t0 = time.time()

        for graphs, labels in train_loader:
            labels = labels.cuda()
            features = graphs.ndata['attr'].cuda()
            outputs = model(graphs, features)

            optimizer.zero_grad()

            loss = loss_fcn(outputs, labels)
            loss.backward()

            optimizer.step()


        dur.append(time.time() - t0)

        print('Average Epoch Time {:.4f}'.format(float(sum(dur)/len(dur))))

        valid_loss, valid_acc = evaluate(model, valid_loader, loss_fcn)
        train_loss, train_acc = evaluate(model, train_loader, loss_fcn)
        print('Train acc {:.4f}'.format(float(train_acc)))
        print('Test acc {:.4f}'.format(float(valid_acc)))

        record[epoch] = (np.mean(dur), train_loss.item(), float(train_acc),
                         valid_loss.item(), float(valid_acc))

        if args.log_norm:
            grad_n = grad_norm(model)
            param_n = param_norm(model)
            grad_record[epoch] = grad_n
            param_record[epoch] = param_n

        scheduler.step()

    return record, grad_record, param_record

def main(args):
    dataset = None
    result_record = {}
    grad_record = {}
    param_record = {}
    set_seed(args.seed)
    if args.cross_validation:
        for fold_idx in range(10):
            args.fold_idx = fold_idx
            dataset, train_loader, valid_loader = task_data(args, dataset)
            model, loss_fcn, optimizer = task_model(args, dataset)
            result_record[args.fold_idx], grad_record[args.fold_idx], param_record[args.fold_idx] = train(args, train_loader, valid_loader, model, loss_fcn, optimizer)
    else:
        dataset, train_loader, valid_loader = task_data(args, dataset)
        model, loss_fcn, optimizer = task_model(args, dataset)
        result_record[args.fold_idx], grad_record[args.fold_idx], param_record[args.fold_idx] = train(args, train_loader, valid_loader, model, loss_fcn, optimizer)

    return result_record, grad_record, param_record

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GIN')

    # 1) general params
    parser.add_argument("--gpu", type=int, default=0,
                        help="gpu")
    parser.add_argument("--seed", type=int, default=9,
                        help='random seed')
    parser.add_argument("--self_loop", action='store_true',
                        help='add self_loop to graph data')
    parser.add_argument("--log_dir", type=str, help='path to the output log file')
    parser.add_argument("--data_dir", type=str, help='path to the datas')

    parser.add_argument('--exp', type=str, help='experiment name')
    parser.add_argument(
        '--dataset', type=str, default='MUTAG',
        choices=['MUTAG', 'PTC', 'NCI1', 'PROTEINS', 'COLLAB', 'IMDBBINARY', 'IMDBMULTI', 'REDDITBINARY', 'REDDITMULTI5K'],
        help='name of dataset (default: MUTAG)'
    )

    # 2) model params
    parser.add_argument("--model", type=str, default='GIN',
                        help='graph models')
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="learning rate")
    parser.add_argument("--dropout", type=float, default=0.5,
                        help='dropout probability')
    parser.add_argument("--epoch", type=int, default=400,
                        help="number of training epochs")
    parser.add_argument("--n_hidden", type=int, default=64,
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
    parser.add_argument('--fold_idx', type=int, default=0,
        help='the index(<10) of fold in 10-fold validation'
    )

    parser.add_argument('--graph_pooling_type', type=str,
        default="sum", choices=["sum", "mean", "max"],
        help='type of graph pooling: sum, mean or max')
    parser.add_argument('--neighbor_pooling_type', type=str,
        default="sum", choices=["sum", "mean", "max"],
        help='type of neighboring pooling: sum, mean or max')
    parser.add_argument('--split_name', type=str,
                        default='fold10', choices=['fold10', 'rand'],
                        help='cross validation split type')
    parser.add_argument('--learn_eps', action="store_true",
        help='learn the epsilon weighting')
    # 3) specific params

    parser.add_argument('--cross_validation', action='store_true',
                        help='Do 10-fold-Cross validation')
    parser.add_argument('--log_norm', action='store_true',
                        help='log normalization information')
    parser.add_argument('--degree_as_label', action='store_true',
                        help='use node degree as node labels')
    parser.add_argument('--norm_type', type=str,
                        default='gn',
                        help='type of normalization')


    args = parser.parse_args()

    os.environ['DGL_DOWNLOAD_DIR'] = args.data_dir

    print(args)
    result, grad, param = main(args)

    def output_result(args, result, grad=None, param=None):
        raw_record = {}
        for seed in result.keys():
            content = result[seed]
            for epoch in content.keys():
                time, train_loss, train_acc, \
                valid_loss, valid_acc = content[epoch]
                if epoch not in raw_record.keys():
                    raw_record[epoch] = {
                        'time': [],
                        'train_loss':[],
                        'train_acc':[],
                        'test_loss':[],
                        'test_acc':[]
                    }

                raw_record[epoch]['time'].append(time)
                raw_record[epoch]['train_loss'].append(train_loss)
                raw_record[epoch]['train_acc'].append(train_acc)
                raw_record[epoch]['test_loss'].append(valid_loss)
                raw_record[epoch]['test_acc'].append(valid_acc)

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

        for epoch in range(len(raw_record.keys())):
            time = np.mean(raw_record[epoch]['time'])
            train_loss = np.mean(raw_record[epoch]['train_loss'])
            train_acc = np.mean(raw_record[epoch]['train_acc'])

            test_loss = np.mean(raw_record[epoch]['test_loss'])
            test_acc = np.mean(raw_record[epoch]['test_acc'])

            train_max_acc = np.max(raw_record[epoch]['train_acc'])
            train_min_acc = np.min(raw_record[epoch]['train_acc'])
            train_std_acc = np.std(raw_record[epoch]['train_acc'])
            test_max_acc = np.max(raw_record[epoch]['test_acc'])
            test_min_acc = np.min(raw_record[epoch]['test_acc'])
            test_std_acc = np.std(raw_record[epoch]['test_acc'])
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

        sheet = f.add_sheet("args")
        sheet.write(0,0,str(args))

        f.save(save_path)


    output_result(args, result, grad, param)



