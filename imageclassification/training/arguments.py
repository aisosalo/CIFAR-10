import argparse

import ast


def aslist(lst):
    return ast.literal_eval(lst)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', default='/data/')
    parser.add_argument('--metadata_root', default='/data/')
    parser.add_argument('--dataset_name', choices=['CIFAR10',
                                                   'CIFAR100',
                                                   ], default='CIFAR10')
    parser.add_argument('--snapshots', default='/snapshots')
    parser.add_argument('--logs', default='/logs')
    parser.add_argument('--train_meta', default='train_meta.csv')
    
    parser.add_argument('--experiment', type=str, choices=['vggbndrop',
                                                           'vgg'
                                                           ], default='vgg')
    
    parser.add_argument('--num_classes', type=int, choices=[10, # CIFAR10
                                                            100 # CIFAR100
                                                            ], default=10)
    
    parser.add_argument('--color_space', type=str, choices=['yuv',
                                                            'rgb'
                                                            ], default='rgb')
    
    parser.add_argument('--optimizer', type=str, choices=['adam',
                                                          'sgd'
                                                          ], default='sgd')
    parser.add_argument('--set_nesterov', default=True) # used with SGD
    parser.add_argument('--learning_rate_decay', type=float, choices=[0.1,
                                                         0.2,  # https://github.com/szagoruyko/wide-residual-networks/blob/master/logs/vgg_24208029/
                                                         ], default=0.1)
    parser.add_argument('--lr', type=float, choices=[1e-4,
                                                     1e-3,
                                                     1e-2,
                                                     1e-1  # https://github.com/szagoruyko/wide-residual-networks/blob/master/logs/vgg_24208029/
                                                     ], default=1e-4)  # small lr will make the training process very slow, large lr may cause overshooting
    parser.add_argument('--lr_drop', type=aslist, choices=[[160, 260]
                                              ], default=[160, 260])
    parser.add_argument('--wd', type=float, choices=[5e-4,  # https://github.com/szagoruyko/wide-residual-networks/blob/master/logs/vgg_24208029/
                                                     1e-4,
                                                     1e-3,
                                                     1e-2
                                                     ], default=1e-4)

    parser.add_argument('--bs', type=int, default=128)
    parser.add_argument('--val_bs', type=int, default=256)

    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument('--fold', type=int, default=-1)

    parser.add_argument('--n_epochs', type=int, choices=[300
                                                         ], default=300)
    parser.add_argument('--n_threads', type=int, choices=[24,
                                                     12,
                                                     6
                                                     ], default=24)

    parser.add_argument('--seed', type=int, default=444)

    args = parser.parse_args()

    return args
