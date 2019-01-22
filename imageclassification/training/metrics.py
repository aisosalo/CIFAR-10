import os
import numpy as np
from termcolor import colored

from sklearn.metrics import confusion_matrix
from imageclassification.kvs import GlobalKVS


def log_metrics(boardlogger, train_loss, val_loss, gt, preds):
    kvs = GlobalKVS()

    # Computing Validation metrics
    cm = confusion_matrix(gt, preds.argmax(1))

    acc = np.mean(cm.diagonal().astype(float) / (cm.sum(axis=1) + 1e-9))

    res = {
     'epoch': kvs['cur_epoch'],
     'val_loss': val_loss,
     'acc': acc,
     }

    print(colored('====> ', 'green') + f'Train loss: {train_loss:.5f}')
    print(colored('====> ', 'green') + f'Validation loss: {val_loss:.5f}')

    boardlogger.add_scalars('Losses', {'train': train_loss, 'val': val_loss}, kvs['cur_epoch'])

    boardlogger.add_scalars('Metrics', {f'Acc_{kvs["args"].experiment}': res['acc']}, kvs['cur_epoch'])

    kvs.update(f'losses_fold_[{kvs["cur_fold"]}]', {'epoch': kvs['cur_epoch'],
                                                    'train_loss': train_loss,
                                                    'val_loss': val_loss,
                                                    'acc': acc})

    kvs.update(f'val_metrics_fold_[{kvs["cur_fold"]}]', res)

    kvs.save_pkl(os.path.join(kvs['args'].snapshots, kvs['args'].dataset_name, kvs['snapshot_name'], 'session.pkl'))
