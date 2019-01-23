"""
BSD 3-Clause License

Copyright (c) Soumith Chintala 2016,
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

https://github.com/pytorch/vision/blob/master/LICENSE#L1-L29

"""


import os
import os.path

import tarfile

import hashlib

from six.moves import urllib


def check_integrity(fpath, md5=None):
    """
    Source: https://github.com/pytorch/vision/blob/master/torchvision/datasets/utils.py#L18-L31

    :param fpath:
    :param md5:
    :return:
    """
    if md5 is None:
        return True
    if not os.path.isfile(fpath):
        return False
    md5o = hashlib.md5()
    with open(fpath, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):  # read in 1MB chunks
            md5o.update(chunk)
    md5c = md5o.hexdigest()
    if md5c != md5:
        return False
    return True


def download_url(url, root, fname, md5):
    """
    Source: https://github.com/pytorch/vision/blob/master/torchvision/datasets/utils.py#L47-L83

    :param url: str, URL to download file from
    :param root: str, directory to place downloaded file in
    :param filename: str, name to save the file under
    :param md5: str, MD5 checksum of the download
    :return:
    """

    root = os.path.expanduser(root)
    fpath = os.path.join(root, fname)

    if os.path.isfile(fpath) and check_integrity(fpath, md5):
        print('Using downloaded and verified file: ' + fpath)
    else:
        try:
            print('Downloading ' + url + ' to ' + fpath)
            urllib.request.urlretrieve(url, fpath)
        except OSError:
            print('Download failed')


def download(dseturl, root, fname, tgz, destination_dir):
    """
    Modified from https://pytorch.org/docs/stable/_modules/torchvision/datasets/cifar.html#CIFAR10

    :param dseturl:
    :param root:
    :param fname:
    :param tgz:
    :param destination_dir:
    :return:
    """
    download_url(dseturl, root, fname, tgz)

    with tarfile.open(os.path.join(root, fname), 'r:gz') as _tar:  # Antti mod according to https://stackoverflow.com/a/47584760
        for entry in _tar:
            if entry.isdir():
                continue
            fn = entry.name.rsplit('/', 1)[1]
            _tar.makefile(entry, os.path.join(root, destination_dir, fn))  # change destination to match 'destination_dir'


def pass_through(path, filelst):
    """
    Source: https://github.com/pytorch/vision/blob/master/torchvision/datasets/cifar.py#L134-L141

    :param root:
    :param base_fldr:
    :param filelst:
    :return:
    """
    for fentry in filelst:
        fname, md5 = fentry[0], fentry[1]
        fpath = os.path.join(path, fname)
        if not check_integrity(fpath, md5):
            return False
    return True

