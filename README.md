# PyTorch Implementation of CIFAR-10 Image Classification Pipeline Using VGG Like Network

We present here our solution to the famous machine learning problem of image classification with [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset with 60000 labeled images. The aim is to learn and assign a category for these `32x32` pixel images.


## Requirements

* Python 3.6
* [tensorboardX](https://tensorboardx.readthedocs.io/en/latest/index.html)


## Dataset

TODO

The CIFAR-10 dataset, as it is provided, consists of 5 batches of training images which sum up to 50000 and a batch of 10000 test images.

Each test batch consists of exactly 1000 randomly-selected images from each class. The training batches contain images in random order, some training batches having more images from one class than another. Together, the training batches contain exactly 5000 images from each class.

Here we have used only the 50000 images originally meant for training. [Stratified K-Folds cross-validation]()is used to spllit the data so that the percentage of samples for each class is preserved.



## Model

We have made a PyTorch implementation of [Sergey Zagoruyko and Nikos Komodakis](https://github.com/szagoruyko/wide-residual-networks) VGG like network for the task.

```
DataParallel(
  (module): VGG(
    (features): Sequential(
      (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (1): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU(inplace)
      (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (4): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
      (5): ReLU(inplace)
      (6): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=True)
      (7): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (8): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
      (9): ReLU(inplace)
      (10): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (11): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
      (12): ReLU(inplace)
      (13): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=True)
      (14): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (15): BatchNorm2d(256, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
      (16): ReLU(inplace)
      (17): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (18): BatchNorm2d(256, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
      (19): ReLU(inplace)
      (20): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (21): BatchNorm2d(256, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
      (22): ReLU(inplace)
      (23): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (24): BatchNorm2d(256, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
      (25): ReLU(inplace)
      (26): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=True)
      (27): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (28): BatchNorm2d(512, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
      (29): ReLU(inplace)
      (30): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (31): BatchNorm2d(512, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
      (32): ReLU(inplace)
      (33): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (34): BatchNorm2d(512, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
      (35): ReLU(inplace)
      (36): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (37): BatchNorm2d(512, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
      (38): ReLU(inplace)
      (39): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=True)
      (40): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (41): BatchNorm2d(512, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
      (42): ReLU(inplace)
      (43): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (44): BatchNorm2d(512, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
      (45): ReLU(inplace)
      (46): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (47): BatchNorm2d(512, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
      (48): ReLU(inplace)
      (49): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (50): BatchNorm2d(512, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
      (51): ReLU(inplace)
    )
    (classifier): Sequential(
      (0): Linear(in_features=512, out_features=10, bias=True)
    )
  )
)
```


## Data Augmentations

In this implementation we only use [horizontal flips](https://mipt-oulu.github.io/solt/transforms.html#solt.transforms.RandomFlip). The images are padded into size `34x34` using [reflective padding](https://mipt-oulu.github.io/solt/transforms.html#solt.transforms.PadTransform) and then crop the images back into size `32x32`. [Random cropping](https://mipt-oulu.github.io/solt/transforms.html#solt.transforms.CropTransform) is used as an augmentation in the training and then [center cropping](https://mipt-oulu.github.io/solt/transforms.html#solt.transforms.CropTransform) in the validation phase.

In their experiments, [Sergey Zagoruyko and Nikos Komodakis](https://github.com/szagoruyko/wide-residual-networks) seem to have used whitened data. -We use the original data with [contrast transformation](https://mipt-oulu.github.io/solt/transforms.html#solt.transforms.PadTransform).-


## Training

In PyCharm

```
$ run_experiments.py --dataset_name CIFAR10 --num_classes 10 --experiment vgg --bs 128 --optimizer sgd --lr 0.1 --wd 5e-4 --learning_rate_decay 0.2 --color_space rgb --set_nesterov True
```

```
$ run_experiments.py --dataset_name CIFAR10 --num_classes 10 --experiment vgg --bs 128 --optimizer adam --lr 0.001 --wd 1e-3 --learning_rate_decay 0.1 --color_space rgb
```


## Results for CIFAR-10

| Network          | Optimizer    | Accuracy (%) | Epochs (#)|
|:----------------:|:------------:|:------------:|:---------:|
| VGG              | SGD          | -            | -         |
| VGG              | Adam         | -            | -         |


## Acknowledgements

[Research Unit of Medical Imaging, Physics and Technology](http://www.mipt-oulu.fi/) is acknowledged for providing the hardware for running the experiments.


## Authors

[Antti Isosalo](https://github.com/aisosalo/) & [Aleksei Tiulpin](https://github.com/lext), [University of Oulu](https://www.oulu.fi/university/), 2018-


## References

### Model Architecture

* Zagoruyko, Sergey, and Nikos Komodakis. "[Wide Residual Networks](https://arxiv.org/abs/1605.07146)." Proceedings of the British Machine Vision Conference (BMVC), 2016.

### Data Augmentation

* Tiulpin, Aleksei, "[Streaming Over Lightweight Data Transformations](https://mipt-oulu.github.io/solt/)." Research Unit of Medical Imaging, Physics and Technology, University of Oulu, Finalnd, 2018.

### Dataset

* Krizhevsky, Alex, and Geoffrey Hinton. "[Learning multiple layers of features from tiny images](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf)." Vol. 1. No. 4. Technical report, University of Toronto, 2009.
