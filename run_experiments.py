import sys
print(sys.version, sys.platform, sys.executable)

from termcolor import colored

from torch.optim.lr_scheduler import MultiStepLR

from imageclassification.kvs import GlobalKVS
from imageclassification.training import utilities
from imageclassification.training import session
from imageclassification.training import dataset as ds
import imageclassification.training.metrics as metrics

import cv2
cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


if __name__ == "__main__":
    kvs = GlobalKVS()

    session.init_session()
    ds.init_metadata()
    dataset, dataset_length = ds.init_dataset(kvs['args'].dataset_root, kvs['args'].dataset_name)

    session.init_data_processing(dataset)
    writers = session.init_folds()

    for fold_id in kvs['cv_split_train']:
        kvs.update('cur_fold', fold_id)
        kvs.update('prev_model', None)
        print(colored('====> ', 'blue') + f'Training fold {fold_id}...')

        train_index, val_index = kvs['cv_split_train'][fold_id]
        train_loader, val_loader = session.init_loaders(dataset, kvs['metadata'].iloc[train_index],
                                                        kvs['metadata'].iloc[val_index])

        net = utilities.init_model()
        print(net)
        optimizer = utilities.init_optimizer(net.parameters())
        print(optimizer)

        print('Learning rate drop schedule: ', kvs['args'].lr_drop)

        scheduler = MultiStepLR(optimizer, milestones=kvs['args'].lr_drop, gamma=kvs['args'].learning_rate_decay)

        for epoch in range(kvs['args'].n_epochs):
            kvs.update('cur_epoch', epoch)

            print(colored('====> ', 'red') + 'Learning rate: ', str(scheduler.get_lr())[1:-1])
            train_loss = utilities.train_epoch(net, optimizer, train_loader)
            val_out = utilities.validate_epoch(net, val_loader)
            val_loss, preds, gt, val_acc = val_out

            print('Epoch: ', epoch)
            print('Acc: ', val_acc)

            metrics.log_metrics(writers[fold_id], train_loss, val_loss, gt, preds)

            session.save_checkpoint(net, 'acc', 'gt')  # gt, greater than
            
            scheduler.step()
