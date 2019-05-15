import argparse
import logging
import os
import time

import numpy as np
import torch
import torch.nn as nn
from data_cleaning import ParkingDataLoader
from torch.utils.data import TensorDataset, DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--network', type=str, choices=['resnet', 'odenet'], default='odenet')
parser.add_argument('--tol', type=float, default=1e-3)
parser.add_argument('--adjoint', type=eval, default=False, choices=[True, False])
parser.add_argument('--downsampling-method', type=str, default='conv', choices=['conv', 'res'])
parser.add_argument('--nepochs', type=int, default=100)
parser.add_argument('--data_aug', type=eval, default=True, choices=[True, False])
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--test_batch_size', type=int, default=256)

parser.add_argument('--save', type=str, default='./experiment1')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()

loader = ParkingDataLoader()

train_data, validation_data, test_data = loader.get_train_validation_test_datasets(validation_split=0.16,
                                                                                   test_split=0.2)

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint


def norm(dim):
    return nn.GroupNorm(min(32, dim), dim)


class ODEfunc(nn.Module):

    def __init__(self, dim):
        super(ODEfunc, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.lin1 = nn.Linear(dim, dim)
        self.lin2 = nn.Linear(dim, dim)
        self.lin3 = nn.Linear(dim, dim)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = self.relu(x)
        out = self.lin1(out)
        out = self.relu(out)
        out = self.lin2(out)
        out = self.relu(out)
        out = self.lin3(out)
        return out


class ODEBlock(nn.Module):

    def __init__(self, odefunc):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0, 1]).float()

    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x)
        out = odeint(self.odefunc, x, self.integration_time, rtol=args.tol, atol=args.tol)
        return out[1]

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value


class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


def get_mnist_loaders(batch_size=128, test_batch_size=256):
    # train_data, validation_data, test_data

    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(train_data.drop('Occupied', axis=1).values.astype(np.float32)),
            torch.tensor(train_data['Occupied'].values.astype(np.float32).reshape((len(train_data), 1)))
        ),
        batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True
    )

    train_eval_loader = DataLoader(
        TensorDataset(
            torch.tensor(validation_data.drop('Occupied', axis=1).values.astype(np.float32)),
            torch.tensor(validation_data['Occupied'].values.astype(np.float32).reshape((len(validation_data), 1)))
        ),
        batch_size=test_batch_size, shuffle=True, num_workers=2, drop_last=True
    )

    test_loader = DataLoader(
        TensorDataset(
            torch.tensor(test_data.drop('Occupied', axis=1).values.astype(np.float32)),
            torch.tensor(test_data['Occupied'].values.astype(np.float32).reshape((len(test_data), 1)))
        ),
        batch_size=test_batch_size, shuffle=True, num_workers=2, drop_last=True
    )

    return train_loader, test_loader, train_eval_loader


def inf_generator(iterable):
    """Allows training with DataLoaders in a single infinite loop:
        for i, (x, y) in enumerate(inf_generator(train_loader)):
    """
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()


def learning_rate_with_decay(batch_size, batch_denom, batches_per_epoch, boundary_epochs, decay_rates):
    initial_learning_rate = args.lr * batch_size / batch_denom

    boundaries = [int(batches_per_epoch * epoch) for epoch in boundary_epochs]
    vals = [initial_learning_rate * decay for decay in decay_rates]

    def learning_rate_fn(itr):
        lt = [itr < b for b in boundaries] + [True]
        i = np.argmax(lt)
        return vals[i]

    return learning_rate_fn


def one_hot(x, K):
    return np.array(x[:, None] == np.arange(K)[None, :], dtype=int)


def accuracy(model, dataset_loader):
    losses = []
    for x, y in dataset_loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        losses.append(np.sqrt(loss.cpu().detach().numpy()))
    return (1 - sum(losses) / len(losses)) * 100


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def get_logger(logpath, filepath, package_files=[], displaying=True, saving=False, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    return logger


def getModel(size=64, layers=1):
    global model
    feature_layers = [ODEBlock(ODEfunc(size)) for _ in range(layers)]
    fc_layers = [nn.Linear(size, 1)]
    model = nn.Sequential(nn.Linear(16, size), *feature_layers, *fc_layers).to(device)
    return model


if __name__ == '__main__':

    logger = get_logger(logpath=os.path.join(args.save, 'logs'), filepath=os.path.abspath(__file__))
    logger.info(args)

    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

    criterion = nn.MSELoss().to(device)

    train_loader, test_loader, train_eval_loader = get_mnist_loaders(
        args.batch_size, args.test_batch_size
    )

    data_gen = inf_generator(train_loader)
    batches_per_epoch = len(train_loader)

    lr_fn = learning_rate_with_decay(
        args.batch_size, batch_denom=128, batches_per_epoch=batches_per_epoch, boundary_epochs=[60, 100, 140],
        decay_rates=[1, 0.1, 0.01, 0.001]
    )

    best_acc = 0
    batch_time_meter = RunningAverageMeter()
    f_nfe_meter = RunningAverageMeter()
    b_nfe_meter = RunningAverageMeter()
    end = time.time()

    for dims in [768, 512, 256, 128, 64, 32]:
        for layers in [12, 10, 8, 6, 4, 2]:
            try:
                model = getModel(dims, layers=layers)
                optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
                logger.info(model)
                logger.info('Number of parameters: {}'.format(count_parameters(model)))
                print(args.nepochs * batches_per_epoch)
                with open("results2", mode="a") as f:
                    f.write("layers: " + str(layers) + "\n")
                    f.write("dims: " + str(dims) + "\n")
                for itr in range(args.nepochs * batches_per_epoch):
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_fn(itr)

                    optimizer.zero_grad()
                    x, y = data_gen.__next__()

                    x = x.to(device)
                    y = y.to(device)
                    logits = model(x)
                    loss = criterion(logits, y)
                    loss.backward()
                    optimizer.step()

                    batch_time_meter.update(time.time() - end)

                    end = time.time()

                    if itr % batches_per_epoch == 0:
                        with torch.no_grad():
                            train_acc = accuracy(model, train_eval_loader)
                            val_acc = accuracy(model, test_loader)
                            if val_acc > best_acc:
                                torch.save({'state_dict': model.state_dict(), 'args': args},
                                           os.path.join(args.save, 'model.pth'))
                                best_acc = val_acc
                            print("------------------------")
                            print("loss:", 100 - np.sqrt(loss.cpu().detach().numpy()) * 100)
                            logger.info(
                                "Epoch {:04d} | Time {:.3f} ({:.3f}) | NFE-F {:.1f} | NFE-B {:.1f} | "
                                "Train Acc {:.10f} | Test Acc {:.10f}".format(
                                    itr // batches_per_epoch, batch_time_meter.val, batch_time_meter.avg,
                                    f_nfe_meter.avg,
                                    b_nfe_meter.avg, train_acc, val_acc
                                )
                            )
                with open("results2", mode="a") as f:
                    f.write("layers: " + str(layers) + "\n")
                    f.write("dims: " + str(dims) + "\n")
                    f.write("Epoch {:04d} | Time {:.3f} ({:.3f}) | NFE-F {:.1f} | NFE-B {:.1f} | "
                            "Train Acc {:.10f} | Test Acc {:.10f} \n".format(0, batch_time_meter.val, batch_time_meter.avg,
                                                                          f_nfe_meter.avg, b_nfe_meter.avg, train_acc,
                                                                          val_acc))
            except Exception as error:
                print(error)
                pass
