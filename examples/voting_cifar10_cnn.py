"""Example on classification using CIFAR-10."""

import argparse
import random
import time
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchensemble.voting import VotingClassifier

from torchensemble.utils.logger import set_logger
from torchensemble.utils.dataloader import load_cifar10
# from torchensemble.utils.optimizers import SGD
# from torchensemble.utils.optimizers import affineSGD


def display_records(records, logger):
    msg = (
        "{:<28} | Training Time: {:.2f} s"
    )
    # msg = (
    #     "{:<28} | Testing Acc: {:.2f} % | Training Time: {:.2f} s |"
    #     " Evaluating Time: {:.2f} s"
    # )

    print("\n")
    for method, training_time in records:
        logger.info(msg.format(method, training_time))
    # for method, training_time, evaluating_time, acc in records:
    #     logger.info(msg.format(method, acc, training_time, evaluating_time))


def set_random_seed(seed):
    """Set a seed for deterministic behaviors.
    Note: If someone runs an experiment with a pre-selected manual seed, he can
    definitely reproduce the results with the same seed; however, if he runs the
    experiment with seed=None and re-run the experiments using the seed previously
    returned from this function (e.g. the returned seed might be logged to
    Tensorboard), and if cudnn is used in the code, then there is no guarantee
    that the results will be reproduced with the recovered seed.
    Args:
        seed (int|None): seed to be used. If None, a default seed based on
            pid and time will be used.
    Returns:
        The seed being used if ``seed`` is None.
    """
    if seed is None:
        seed = int(np.uint32(hash(str(os.getpid()) + '|' + str(time.time()))))
    else:
        # torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # torch.use_deterministic_algorithms(True)
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.gelu(self.conv1(x)))
        x = self.pool(F.gelu(self.conv2(x)))
        x = x.view(-1, 400)
        x = F.gelu(self.fc1(x))
        x = F.gelu(self.fc2(x))
        x = self.fc3(x)
        return x


parser = argparse.ArgumentParser(description='CIFAR10 Ensemble Training')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--wd', default=5e-4, type=float, help='weight decay')
parser.add_argument('--bs', default=128, type=int)
parser.add_argument('--n', default=10, type=int, help='ensemble size')
parser.add_argument('--nj', default=10, type=int, help='num of jobs')
parser.add_argument('--model', default='lenet', type=str)
parser.add_argument('--bias', action='store_false')
parser.add_argument('--fullrank', action='store_true')
parser.add_argument('--fixedRV', action='store_true')
parser.add_argument('--scale', default=1.0, type=float)
parser.add_argument('--weightonly', action='store_true')
# parser.add_argument('--samenorm', action='store_true')
# parser.add_argument('--norm', action='store_true')
parser.add_argument('--diag', action='store_true')
parser.add_argument('--exp', default=None, type=float)
parser.add_argument('--optimizer', default='sgd', type=str)
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--log_dir', default='runs', type=str)
parser.add_argument('--epoch', default=200, type=int)
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--extra', default=None, type=str)

args = parser.parse_args()

# set random seed
seed = set_random_seed(args.seed)


if __name__ == "__main__":

    # set logger dir
    if args.fullrank:
        str_fullrank = '_fullrank'
    else:
        str_fullrank = ''
    if args.fixedRV:
        str_rv = '_fixedRV'
    else:
        str_rv = ''
    if args.weightonly:
        str_wo = '_weightonly'
    else:
        str_wo = ''
    # if args.samenorm:
    #     str_sn = '_samenorm'
    # else:
    #     str_sn = ''
    # if args.norm:
    #     str_norm = '_norm'
    # else:
    #     str_norm = ''
    if args.diag:
        str_diag = '_diag'
    else:
        str_diag = ''
    if args.exp:
        str_exp = f'_exp{args.exp}'
    else:
        str_exp = ''
    if args.extra is None:
        str_extra = ''
    else:
        str_extra = '_' + args.extra
    log_dir = os.path.join(args.dataset, args.optimizer)
    # log_file = ('%s_epoch%d_n%d_nj%d_seed%d_scale%.1f%s%s%s%s%s%s' % (
    #             args.model, args.epoch, args.n, args.nj, args.seed, args.scale,
    #             str_fullrank, str_rv, str_wo, str_diag, str_exp, str_extra))
    log_file = f"{args.model}_epoch{args.epoch}_n{args.n}_nj{args.nj}_seed{args.seed}" \
               f"_lr{args.lr}_scale{args.scale}{str_fullrank}{str_rv}{str_wo}" \
               f"{str_diag}{str_exp}{str_extra}"

    # Hyper-parameters
    # n_estimators = 10
    # batch_size = 128
    # optimizer = 'SGD'
    # lr = 1e-3
    # weight_decay = 5e-4
    # epochs = 100
    ood = True
    if ood:
        estimator_args = dict(num_classes=6)
    else:
        estimator_args = None

    # Utils
    data_dir = "../../Dataset/cifar"  # MODIFY THIS IF YOU WANT
    records = []
    # torch.manual_seed(0)

    # Load data
    train_loader, test_loader, ood_loader = load_cifar10(
        data_dir, batch_size=args.bs, split=ood)

    logger = set_logger(log_dir, log_file, use_tb_logger=True)

    # VotingClassifier
    model = VotingClassifier(
        estimator=LeNet5, n_estimators=args.n, estimator_args=estimator_args, 
        cuda=True, n_jobs=args.nj
    )

    # Set the optimizer
    if args.optimizer == 'sgd':
        optimizer_name = 'SGD'
        extra_kwargs = dict(lr=args.lr, weight_decay=args.wd)
    elif args.optimizer == 'gtk':
        optimizer_name = 'affineSGD'
        extra_kwargs = dict(lr=args.lr, weight_decay=args.wd,
                            fullrank=args.fullrank, fixed_rand_vec=args.fixedRV,
                            weight_only=args.weightonly, diag=args.diag,
                            scale=args.scale, exponential=args.exp)
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")
    model.set_optimizer(optimizer_name, **extra_kwargs)

    # Training
    tic = time.time()
    model.fit(train_loader,
              epochs=args.epoch,
              test_loader=test_loader,
              ood_loader=ood_loader,
              save_model=False)
    toc = time.time()
    training_time = toc - tic

    # # Evaluating
    # tic = time.time()
    # testing_acc = model.evaluate(test_loader)
    # toc = time.time()
    # evaluating_time = toc - tic
    #
    # records.append(
    #     ("VotingClassifier", training_time, evaluating_time, testing_acc)
    # )

    records.append(
        ("VotingClassifier", training_time)
    )

    # Print results on different ensemble methods
    display_records(records, logger)
