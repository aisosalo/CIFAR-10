from tqdm import tqdm

import gc

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch import optim

from imageclassification.kvs import GlobalKVS
import imageclassification.training.model as mdl


def init_model():
    kvs = GlobalKVS()
    net = mdl.get_model(kvs['args'].experiment, kvs['args'].num_classes)

    if kvs['gpus'] > 1:
        net = nn.DataParallel(net)  #.to('cuda')

    net = net.to('cuda')
    return net


def init_optimizer(parameters):
    kvs = GlobalKVS()
    if kvs['args'].optimizer == 'adam':
        return optim.Adam(parameters, lr=kvs['args'].lr, weight_decay=kvs['args'].wd)
    elif kvs['args'].optimizer == 'sgd':
        return optim.SGD(parameters, lr=kvs['args'].lr, weight_decay=kvs['args'].wd, momentum=0.9, nesterov=kvs['args'].set_nesterov)
    else:
        raise NotImplementedError


def train_epoch(net, optimizer, train_loader):
    kvs = GlobalKVS()
    net.train(True)

    running_loss = 0.0
    n_batches = len(train_loader)

    epoch = kvs['cur_epoch']
    max_ep = kvs['args'].n_epochs

    device = next(net.parameters()).device

    pbar = tqdm(total=n_batches)  # the number of expected iterations here is n_batches

    for i, batch in enumerate(train_loader):
        optimizer.zero_grad()  # set the gradients to zero before starting backpropragation, PyTorch accumulates the gradients on subsequent backward passes

        labels = batch['label'].long().to(device)
        inputs = batch['img'].to(device)

        outputs = net(inputs)

        loss = F.cross_entropy(outputs, labels)

        loss.backward()  # accumulates the gradient (by addition) for each parameter
        optimizer.step()  # performs a parameter update based on the current gradient
        running_loss += loss.item()  # adds to running_loss the scalar value held in the loss
        # loss represents the sum of losses over all examples in the batch

        gc.collect()
        pbar.set_description(
            f'[{epoch+1} | {max_ep}] Train loss: {running_loss / (i + 1):.3f} / Loss {loss.item():.3f}')
        pbar.update()

    gc.collect()
    pbar.close()

    # train_loss
    return running_loss / n_batches


def validate_epoch(net, test_loader):
    kvs = GlobalKVS()
    net.eval()

    running_loss = 0.0
    n_batches = len(test_loader)
    epoch = kvs['cur_epoch']
    max_ep = kvs['args'].n_epochs

    device = next(net.parameters()).device

    probs_lst = []
    gt_lst = []

    pbar = tqdm(total=n_batches)

    correct = 0
    all_samples = 0
    with torch.no_grad():  # stop autograd from tracking history on Tensors; autograd records computation history on the fly to calculate gradients later
        for i, batch in enumerate(test_loader):
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
                f'[{epoch+1} | {max_ep}] Validation accuracy: {100. * correct / all_samples:.0f}%')
            pbar.update()

        gc.collect()
        pbar.close()

    # val_loss, preds, gt, val_acc
    return running_loss / n_batches, np.array(probs_lst), np.array(gt_lst), correct / all_samples
