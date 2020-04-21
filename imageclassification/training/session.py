import os
import time
import random

import numpy as np

from termcolor import colored

from functools import partial

from tensorboardX import SummaryWriter

import torch
from torch.utils.data import DataLoader

from torchvision import transforms as tv_transforms

import solt.transforms as slt
import solt.core as slc

import operator

from imageclassification.training.arguments import parse_args
from imageclassification.training.dataset import ImageClassificationDataset
from imageclassification.training.dataset import apply_by_index, img_labels2solt, unpack_solt_data
from imageclassification.training.transformations import init_train_augs
from imageclassification.kvs import GlobalKVS
import imageclassification.training.transformations as trnsfs

PAD_TO = 34
CROP_SIZE = 32

DEBUG = sys.gettrace() is not None


def init_session():
    if not torch.cuda.is_available():
        raise EnvironmentError('The code must be run on GPU.')

    kvs = GlobalKVS()

    args = parse_args()
    
    if DEBUG:
        args.n_threads = 0

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    snapshot_name = time.strftime('%Y_%m_%d_%H_%M')
    os.makedirs(os.path.join(args.snapshots, args.dataset_name, snapshot_name), exist_ok=True)

    kvs.update('pytorch_version', torch.__version__)
    print('Pytorch version: ', torch.__version__)

    if torch.cuda.is_available():
        kvs.update('cuda', torch.version.cuda)
        kvs.update('gpus', torch.cuda.device_count())
        print('CUDA version: ', torch.version.cuda)
    else:
        kvs.update('cuda', None)
        kvs.update('gpus', None)

    kvs.update('snapshot_name', snapshot_name)
    kvs.update('args', args)
    kvs.save_pkl(os.path.join(args.snapshots, args.dataset_name, snapshot_name, 'session.pkl'))

    return args, snapshot_name


def init_data_processing(ds):
    kvs = GlobalKVS()

    train_augs = init_train_augs(crop_mode='r', pad_mode='r')  # random crop, reflective padding

    dataset = ImageClassificationDataset(ds, split=kvs['metadata'], color_space=kvs['args'].color_space, transformations=train_augs)

    mean_vector, std_vector = trnsfs.init_mean_std(dataset=dataset,
                                                   batch_size=kvs['args'].bs,
                                                   n_threads=kvs['args'].n_threads,
                                                   save_mean_std=kvs['args'].snapshots + '/' + kvs['args'].dataset_name,
                                                   color_space=kvs['args'].color_space)

    print('Color space: ', kvs['args'].color_space)

    print(colored('====> ', 'red') + 'Mean:', mean_vector)
    print(colored('====> ', 'red') + 'Std:', std_vector)

    norm_trf = tv_transforms.Normalize(torch.from_numpy(mean_vector).float(),
                                       torch.from_numpy(std_vector).float())

    train_trf = tv_transforms.Compose([
        train_augs,
        partial(apply_by_index, transform=norm_trf, idx=0)
    ])

    val_trf = tv_transforms.Compose([
        img_labels2solt,
        slc.Stream([
            slt.PadTransform(pad_to=(PAD_TO, PAD_TO)),
            slt.CropTransform(crop_size=(CROP_SIZE, CROP_SIZE), crop_mode='c'),  # center crop
        ]),
        unpack_solt_data,
        partial(apply_by_index, transform=tv_transforms.ToTensor(), idx=0),
        partial(apply_by_index, transform=norm_trf, idx=0)
    ])

    kvs.update('train_trf', train_trf)
    kvs.update('val_trf', val_trf)
    kvs.save_pkl(os.path.join(kvs['args'].snapshots, kvs['args'].dataset_name, kvs['snapshot_name'], 'session.pkl'))


def init_loaders(dataset, x_train, x_val):
    kvs = GlobalKVS()

    train_dataset = ImageClassificationDataset(dataset,
                                         split=x_train,
                                         color_space=kvs['args'].color_space,
                                         transformations=kvs['train_trf'])

    val_dataset = ImageClassificationDataset(dataset,
                                       split=x_val,
                                       color_space=kvs['args'].color_space,
                                       transformations=kvs['val_trf'])

    train_loader = DataLoader(train_dataset,
                              batch_size=kvs['args'].bs,
                              num_workers=kvs['args'].n_threads,
                              drop_last=True,
                              worker_init_fn=lambda wid: np.random.seed(np.uint32(torch.initial_seed() + wid)))

    val_loader = DataLoader(val_dataset,
                            batch_size=kvs['args'].val_bs,
                            num_workers=kvs['args'].n_threads)

    return train_loader, val_loader


def init_folds():
    kvs = GlobalKVS()
    writers = {}
    cv_split_train = {}
    for fold_id, split in enumerate(kvs['cv_split_all_folds']):
        if kvs['args'].fold != -1 and fold_id != kvs['args'].fold:
            continue
        kvs.update(f'losses_fold_[{fold_id}]', None, list)
        kvs.update(f'val_metrics_fold_[{fold_id}]', None, list)
        cv_split_train[fold_id] = split
        writers[fold_id] = SummaryWriter(os.path.join(kvs['args'].logs,
                                                      kvs['args'].dataset_name,
                                                      'fold_{}'.format(fold_id), kvs['snapshot_name']))

    kvs.update('cv_split_train', cv_split_train)
    kvs.save_pkl(os.path.join(kvs['args'].snapshots, kvs['args'].dataset_name, kvs['snapshot_name'], 'session.pkl'))

    return writers


def save_checkpoint(model, val_metric_name, comparator='lt'): # lt, less than
    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    kvs = GlobalKVS()
    fold_id = kvs['cur_fold']
    epoch = kvs['cur_epoch']
    val_metric = kvs[f'val_metrics_fold_[{fold_id}]'][-1][0][val_metric_name]
    comparator = getattr(operator, comparator)
    cur_snapshot_name = os.path.join(kvs['args'].snapshots, kvs['args'].dataset_name, kvs['snapshot_name'],
                                     f'fold_{fold_id}_epoch_{epoch+1}.pth')

    if kvs['prev_model'] is None:
        print(colored('====> ', 'red') + 'Snapshot was saved to', cur_snapshot_name)
        torch.save(model.state_dict(), cur_snapshot_name)
        kvs.update('prev_model', cur_snapshot_name)
        kvs.update('best_val_metric', val_metric)

    else:
        if comparator(val_metric, kvs['best_val_metric']):
            print(colored('====> ', 'red') + 'Snapshot was saved to', cur_snapshot_name)
            os.remove(kvs['prev_model'])
            torch.save(model.state_dict(), cur_snapshot_name)
            kvs.update('prev_model', cur_snapshot_name)
            kvs.update('best_val_metric', val_metric)

    kvs.save_pkl(os.path.join(kvs['args'].snapshots, kvs['args'].dataset_name, kvs['snapshot_name'], 'session.pkl'))

