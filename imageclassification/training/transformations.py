import os

from termcolor import colored

from functools import partial

import numpy as np

from torch.utils.data import DataLoader

from torchvision import transforms

import solt.transforms as slt
import solt.core as slc

from imageclassification.training.dataset import apply_by_index, img_labels2solt, unpack_solt_data

PAD_TO = 34
CROP_SIZE = 32


def init_mean_std(dataset, batch_size, n_threads, save_mean_std, color_space='rgb'):
    if 'yuv' in color_space:
        filename = 'mean_std_yuv.npy'
    elif 'rgb' in color_space:
        filename = 'mean_std.npy'
    else:
        raise NotImplementedError

    if os.path.isfile(os.path.join(save_mean_std, filename)):
        tmp = np.load(os.path.join(save_mean_std, filename))
        mean_vector, std_vector = tmp
    else:
        tmp_loader = DataLoader(dataset, batch_size=batch_size, num_workers=n_threads)
        mean_vector = None
        std_vector = None
        print(colored('==> ', 'green') + 'Calculating Mean and Standard Deviation')

        for batch in tmp_loader:
            imgs = batch['img']
            if mean_vector is None:
                mean_vector = np.zeros(imgs.size(1))
                std_vector = np.zeros(imgs.size(1))
            for j in range(mean_vector.shape[0]):
                mean_vector[j] += imgs[:, j, :, :].mean()
                std_vector[j] += imgs[:, j, :, :].std()

        mean_vector /= len(tmp_loader)
        std_vector /= len(tmp_loader)

        np.save(os.path.join(save_mean_std, filename),
                [mean_vector.astype(np.float32), std_vector.astype(np.float32)])

    return mean_vector, std_vector


def init_train_augs(crop_mode='r', pad_mode='r'):
    trf = transforms.Compose([
        img_labels2solt,
        slc.Stream([
            slt.PadTransform(pad_to=(PAD_TO, PAD_TO)),
            slt.RandomFlip(p=0.5, axis=1),  # horizontal flip
            slt.CropTransform(crop_size=(CROP_SIZE, CROP_SIZE), crop_mode=crop_mode),
        ], padding=pad_mode),
        unpack_solt_data,
        partial(apply_by_index, transform=transforms.ToTensor(), idx=0),
    ])
    return trf
