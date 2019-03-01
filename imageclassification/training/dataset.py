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
    """
    Source: https://www.cs.toronto.edu/~kriz/cifar.html

    :param file: Python object produced with cPickle
    :return: dictionary
    """
    with open(file, 'rb') as fo:
        cifar_dict = pickle.load(fo, encoding='bytes')

    return cifar_dict


def init_dataset(path, dataset, batch='train'):  # path = kvs['args'].dataset_root
    if 'CIFAR10' == dataset:
        if 'train' == batch:
            batch_list = [
                ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
                ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
                ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
                ['data_batch_4', '634d18415352ddfa80567beed471001a'],
                ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
            ]
        elif 'test' == batch:
            batch_list = [
                ['test_batch', '40351d587109b95175f43aff81a1287e'],
            ]
        else:
            raise NotImplementedError
    elif 'CIFAR100' == dataset:
        if 'train' == batch:
            batch_list = [
                ['train', '16019d7e3df5f24257cddd939b257f8d'],
            ]
        elif 'test' == batch:
            batch_list = [
                ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
            ]
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    path = os.path.join(path, dataset)

    ds = []
    for entry in batch_list:
        print(colored('====> ', 'blue') + 'Processing file: ', os.path.join(path, entry[0]))
        batch = unpickle(os.path.join(path, entry[0]))
        tmp = batch[b'data']
        ds.append(tmp)

    ds = np.concatenate(ds)
    l_dataset = len(ds)
    ds = ds.reshape(l_dataset, 3, 32, 32).swapaxes(1, 3).swapaxes(1, 2)

    return ds, l_dataset

def init_metadata():
    kvs = GlobalKVS()

    meta = pd.read_csv(os.path.join(kvs['args'].metadata_root, kvs['args'].dataset_name, kvs['args'].train_meta'))

    print(f'Dataset (form CSV file) has {meta.shape[0]} entries')

    kvs.update('metadata', meta)

    skf = StratifiedKFold(n_splits=kvs['args'].n_folds)  # this cross-validation object is a variation of KFold that returns stratified folds, preserving the percentage of samples for each class

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
    """
    Applies callable to certain objects in iterable using given indices.
    Parameters

    :param items: tuple or list
    :param transform: callable
    :param idx: int or tuple or or list None
    :return: tuple
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
