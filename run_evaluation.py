import sys
print(sys.version, sys.platform, sys.executable)

import argparse
import random
import os
import pickle
import gc

from functools import partial

import numpy as np

import pandas as pd

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from torchvision import transforms as tv_transforms

from sklearn.metrics import confusion_matrix

from tqdm import tqdm

import solt.transforms as slt
import solt.core as slc

from imageclassification.training.dataset import apply_by_index, img_labels2solt, unpack_solt_data
import imageclassification.training.model as mdl
from imageclassification.training.dataset import ImageClassificationDataset, init_dataset

import cv2
cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

PAD_TO = 34
CROP_SIZE = 32

DEBUG = sys.gettrace() is not None
print('Debug: ', DEBUG)


def ev(net, loader):
    net.eval()

    running_loss = 0.0
    n_batches = len(loader)

    device = next(net.parameters()).device

    probs_lst = []
    gt_lst = []
    net.eval()

    pbar = tqdm(total=n_batches)

    correct = 0
    all_samples = 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            labels = batch['label'].long().to(device)
            inputs = batch['img'].to(device)

            outputs = net(inputs)

            loss = F.cross_entropy(outputs, labels)

            probs_batch = F.softmax(outputs, 1).data.to('cpu').numpy()
            gt_batch = batch['label'].numpy()

            probs_lst.extend(probs_batch.tolist())
            gt_lst.extend(gt_batch.tolist())

            running_loss += loss.item()

            pred = np.array(probs_lst).argmax(1)
            correct += np.equal(pred, np.array(gt_lst)).sum()
            all_samples += len(np.array(gt_lst))

            gc.collect()
            pbar.set_description(
                f'Evaluation accuracy: {100. * correct / all_samples:.0f}%')
            pbar.update()

        gc.collect()
        pbar.close()

    # val_loss, preds, gt, val_acc
    return running_loss / n_batches, np.array(probs_lst), np.array(gt_lst), correct / all_samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', default='/data/')
    parser.add_argument('--metadata_root', default='/data/')
    parser.add_argument('--dataset_name', choices=['CIFAR10',
                                                   'CIFAR100',
                                                   ], default='CIFAR10')
    parser.add_argument('--snapshots', default='/snapshots/CIFAR10')
    parser.add_argument('--snapshot', default='')

    args = parser.parse_args()

    with open(os.path.join(args.snapshots, args.snapshot, 'session.pkl'), 'rb') as f:
        args_snp = pickle.load(f)
        previous_model = args_snp['prev_model'][0]
        args_snp = args_snp['args']
        args_snp = args_snp[0]
        args_snp.snapshots = args.snapshots
        args_snp.snapshot = args.snapshot
        args = args_snp

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.color_space == 'rgb':
        mean_vector, std_vector = np.load(os.path.join(args.snapshots, 'mean_std.npy'))
    elif args.color_space == 'yuv':
        mean_vector, std_vector = np.load(os.path.join(args.snapshots, 'mean_std_yuv.npy'))
    else:
        raise NotImplementedError

    norm_trf = tv_transforms.Normalize(torch.from_numpy(mean_vector).float(), torch.from_numpy(std_vector).float())

    eval_trf = tv_transforms.Compose([
        img_labels2solt,
        slc.Stream([
            slt.PadTransform(pad_to=(PAD_TO, PAD_TO)),
            slt.CropTransform(crop_size=(CROP_SIZE, CROP_SIZE), crop_mode='c'),  # center crop
        ]),
        unpack_solt_data,
        partial(apply_by_index, transform=tv_transforms.ToTensor(), idx=0),
        partial(apply_by_index, transform=norm_trf, idx=0)
        ])

    dataset, dataset_length = init_dataset(args.dataset_root, args.dataset_name, batch='test')

    metadata = pd.read_csv(os.path.join(args.metadata_root, args.dataset_name, args.dataset_name + '_test_filenames.csv'))

    eval_dataset = ImageClassificationDataset(dataset, metadata, args.color_space, eval_trf)

    eval_loader = DataLoader(eval_dataset, batch_size=10000, num_workers=1)

    net = mdl.get_model(args.experiment, args.num_classes)
    net.load_state_dict(torch.load(previous_model, map_location=lambda storage, location: storage))
    net.eval()
    net = net.to('cuda')

    eval_out = ev(net, eval_loader)
    eval_loss, preds, gt, eval_acc = eval_out

    cm = confusion_matrix(gt, preds.argmax(1))
    print('Confusion Matric: ', cm)

    acc = np.mean(cm.diagonal().astype(float) / (cm.sum(axis=1) + 1e-9))
    print('Acc: ', acc)
