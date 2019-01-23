import sys
print(sys.version, sys.platform, sys.executable)

import pickle
import os.path

import argparse

import pandas as pd

import numpy as np

from imageclassification.utils import download, pass_through


def unpickle(file):
    """
    Source: https://www.cs.toronto.edu/~kriz/cifar.html

    :param file: Python object produced with cPickle
    :return: dictionary

    """
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')

    return dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root',
                        default='/data/')  # working dir
    parser.add_argument('--metadata_root',
                        default='/data/')  # working dir

    parser.add_argument('--dataset', choices=['CIFAR10',
                                              'CIFAR100',
                                             ], default='CIFAR10')

    parser.add_argument('--num_workers', type=int, default=12)
    parser.add_argument('--seed', type=int, default=2222)
    args = parser.parse_args()

    # Create folders
    os.makedirs(os.path.join(args.dataset_root), exist_ok=True)
    os.makedirs(os.path.join(args.dataset_root, args.dataset + '/'), exist_ok=True)
    os.makedirs(args.metadata_root, exist_ok=True)

    # Set path
    path = args.dataset_root + args.dataset + '/'

    # Set dataset related variables
    if 'CIFAR10' == args.dataset:
        url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
        filename = "cifar-10-python.tar.gz"
        tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
        train_list = [
            ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
            ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
            ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
            ['data_batch_4', '634d18415352ddfa80567beed471001a'],
            ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
        ]
        test_list = [
            ['test_batch', '40351d587109b95175f43aff81a1287e'],
        ]
    elif 'CIFAR100' == args.dataset:
        url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
        filename = "cifar-100-python.tar.gz"
        tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
        train_list = [
            ['train', '16019d7e3df5f24257cddd939b257f8d'],
        ]
        test_list = [
            ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
        ]
    else:
        raise NotImplementedError

    # Download and check
    if pass_through(path, train_list + test_list):
        print('Files already downloaded and verified.')
    else:
        download(url, args.dataset_root, filename, tgz_md5, args.dataset + '/')

    # Create metadata
    print('Building metadata...')
    if 'CIFAR10' == args.dataset:
        data = []
        fn = []
        y = []
        for entry in train_list:
            print('Processing file: ', entry[0])
            data = unpickle(os.path.join(path, entry[0]))
            lbl = data[b'labels']
            fname = data[b'filenames']
            y.append(lbl)
            fn.append(fname)

        fn_cat = np.concatenate(fn)
        y_cat = np.concatenate(y)
        df = pd.concat([pd.DataFrame((x.decode('UTF8') for x in fn_cat), columns=["Filename"]),
                        pd.DataFrame(y_cat, columns=["Label"]),
                        pd.DataFrame(list(range(1, len(y_cat) + 1)), columns=["ID"])
                        ], axis=1)
        if not len(df) == 50000:
            raise ValueError('Something went wrong with CIFAR-10 dataset.')
    elif 'CIFAR100' == args.dataset:
        data = []
        fn = []
        y_fine = []
        y_coarse = []
        for entry in train_list:
            print('Processing file: ', entry[0])
            data = unpickle(os.path.join(path, entry[0]))
            lbl = data[b'fine_labels']
            g = data[b'coarse_labels']
            fname = data[b'filenames']
            y_fine.append(lbl)
            y_coarse.append(g)
            fn.append(fname)

        fn_cat = np.concatenate(fn)
        y_fine_cat = np.concatenate(y_fine)
        y_coarse_cat = np.concatenate(y_coarse)
        df = pd.concat([pd.DataFrame((x.decode('UTF8') for x in fn_cat), columns=["Filename"]),
                        pd.DataFrame(y_fine_cat, columns=["Label"]),
                        pd.DataFrame(y_coarse_cat, columns=["Group"]),
                        pd.DataFrame(list(range(1, len(y_fine_cat) + 1)), columns=["ID"])
                        ], axis=1)
        if not len(df) == 50000:
            raise ValueError('Something went wrong with CIFAR-100 dataset.')
    else:
        raise NotImplementedError

    if not os.path.isfile(args.metadata_root + args.dataset + '/train_meta.csv'):
        df.to_csv(args.metadata_root + args.dataset + '/train_meta.csv', sep=',', index=False)
    else:
        print('Metadata already exists.')
