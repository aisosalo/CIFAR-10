import os
import pickle
import copy

from termcolor import colored

import torch
import torch.utils.data as data

from sklearn.model_selection import StratifiedKFold

import numpy as np

import pandas as pd

import solt.data as sld

import cv2

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

from imageclassification.kvs import GlobalKVS


class ImageClassificationDataset(data.Dataset):
    def __init__(self, dataset, split, color_space='rgb', transformations=None):
        self.dataset = dataset
        self.split = split
        self.color_space = color_space
        self.transformations = transformations

    def __getitem__(self, ind):
        if isinstance(ind, torch.Tensor):
            ind = ind.item()

        entry = self.split.iloc[ind]
        indx = entry.ID-1 # ID is a row number starting from 1
        
        dimg = self.dataset[indx, :, :, :]

        if 'yuv' in self.color_space:
            dimg = cv2.cvtColor(dimg, cv2.COLOR_BGR2YUV)
            dimg[:, :, 0] = cv2.equalizeHist(dimg[:, :, 0])

        img, label = self.transformations((entry.Label, dimg))

        res = {'img': img,
               'label': label,
               }

        return res

    def __len__(self):
        return self.split.shape[0]


def unpickle(file):
    '''
    https://www.cs.toronto.edu/~kriz/cifar.html
    '''
    with open(file, 'rb') as fo:
        cifar_dict = pickle.load(fo, encoding='bytes')

    return cifar_dict


def init_dataset(path, dataset):  # path = kvs['args'].dataset_root
    path = os.path.join(path, dataset)
    files = os.listdir(path)
    files = sorted(files)
    ds = []
    for file in files:
        if file.startswith('data'):
            print(colored('====> ', 'blue') + 'Processing file: ', os.path.join(path, file))
            batch = unpickle(os.path.join(path, file))
            tmp = batch[b'data']
            ds.append(tmp)

    ds = np.concatenate(ds)
    ds_len = len(ds)
    ds = ds.reshape(ds_len, 3, 32, 32).swapaxes(1, 3).swapaxes(1, 2)

    return ds, ds_len


def init_metadata():
    kvs = GlobalKVS()

    meta = pd.read_csv(os.path.join(kvs['args'].metadata_root, kvs['args'].dataset_name, kvs['args'].dataset_name + '_train_filenames.csv'))

    print(f'Dataset (form CSV file) has {meta.shape[0]} entries')

    kvs.update('metadata', meta)

    skf = StratifiedKFold(n_splits=kvs['args'].n_folds)  # This cross-validation object is a variation of KFold that returns stratified folds, preserving the percentage of samples for each class

    if 'CIFAR10' in kvs['args'].dataset_name:
        cv_split = [x for x in skf.split(kvs['metadata']['Filename'].astype(str),
                                         kvs['metadata']['Label'],
                                         kvs['metadata']['ID'])]
    elif 'CIFAR100' in kvs['args'].dataset_name:
        cv_split = [x for x in skf.split(kvs['metadata']['Filename'].astype(str),
                                         kvs['metadata']['Label'],
                                         kvs['metadata']['Group'],
                                         kvs['metadata']['ID'])]
    else:
        raise NotImplementedError

    kvs.update('cv_split_all_folds', cv_split)
    kvs.save_pkl(os.path.join(kvs['args'].snapshots, kvs['args'].dataset_name, kvs['snapshot_name'], 'session.pkl'))


def img_labels2solt(inp):
    label, img = inp
    return sld.DataContainer((img, label), fmt='IL')


def unpack_solt_data(dc: sld.DataContainer):
    return dc.data


def apply_by_index(items, transform, idx=0):
    """Applies callable to certain objects in iterable using given indices.
    Parameters
    ----------
    items: tuple or list
    transform: callable
    idx: int or tuple or or list None
    Returns
    -------
    result: tuple
    """
    if idx is None:
        return items
    if not isinstance(items, (tuple, list)):
        raise TypeError
    if not isinstance(idx, (int, tuple, list)):
        raise TypeError

    if isinstance(idx, int):
        idx = [idx, ]

    idx = set(idx)
    res = []
    for i, item in enumerate(items):
        if i in idx:
            res.append(transform(item))
        else:
            res.append(copy.deepcopy(item))

    return res
