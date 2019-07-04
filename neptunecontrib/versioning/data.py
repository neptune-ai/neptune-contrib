#
# Copyright (c) 2019, Neptune Labs Sp. z o.o.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import os
import hashlib

import boto3
import neptune
import numpy as np
import matplotlib.pyplot as plt

from neptunecontrib.monitoring.utils import send_figure


def log_data_version(path, prefix='', experiment=None):
    """Logs data version of file or folder to Neptune

    For a path it calculates the hash and logs it along with the path itself as a property to Neptune experiment.
    Path to dataset can be a file or directory.

    Args:
        path(str): path to the file or directory,
        prefix(str): Prefix that will be added before 'ata_version' and 'data_path'
        experiment(neptune.experiemnts.Experiment or None): if the data should be logged to a particular
           neptune experiment it can be passed here. By default it is logged to the current experiment.

    Examples:
        Initialize Neptune::

            import neptune
            from neptunecontrib.versioning.data import log_data_version
            neptune.init('USER_NAME/PROJECT_NAME')

        Log data version from filepath::

            FILEPATH = '/path/to/data/my_data.csv'
            with neptune.create_experiment():
                log_data_version(FILEPATH)

    """

    _exp = experiment if experiment else neptune

    _exp.set_property('{}data_path'.format(prefix), path)
    _exp.set_property('{}data_version'.format(prefix), _md5_hash_path(path))


def log_s3_data_version(bucket_name, path, prefix='', experiment=None):
    """Logs data version of s3 bucket to Neptune

    For a bucket and path it calculates the hash and logs it along with the path itself as a property to
    Neptune experiment.
    Path is either the s3 bucket key to a file or the begining of a key (in case you use a "folder" structure).

    Args:
        bucket_name(str): name of the s3 bucket
        path(str): path to the file or directory on s3 bucket
        prefix(str): Prefix that will be added before 'data_version' and 'data_path'
        experiment(neptune.experiemnts.Experiment or None): if the data should be logged to a particular
           neptune experiment it can be passed here. By default it is logged to the current experiment.

    Examples:
        Initialize Neptune::

            import neptune
            from neptunecontrib.versioning.data import log_s3_data_version
            neptune.init('USER_NAME/PROJECT_NAME')

        Log data version from bucket::

            BUCKET = 'my-bucket'
            PATH = 'train_dir/'
            with neptune.create_experiment():
                log_s3_data_version(BUCKET, PATH)

    """

    _exp = experiment if experiment else neptune

    _exp.set_property('{}data_path'.format(prefix), '{}/{}'.format(bucket_name, path))
    _exp.set_property('{}data_version'.format(prefix), _md5_hash_bucket(bucket_name, path))


def log_image_dir_snapshots(image_dir, channel_name='image_dir_snapshots', experiment=None, sample=16, seed=1234):
    """Logs visual snapshot of the directory with image data to Neptune.

    For a given directory with images it logs a sample of images as figure to Neptune.
    If the `image_dir` specified contains multiple folders it will sample per folder and create
    multiple figures naming each figure with the folder name.
    See snapshots per class here https://ui.neptune.ml/jakub-czakon/examples/e/EX-95/channels.

    Args:
        image_dir(str): path to directory with images.
        sample(int): number of images that should be sampled for plotting.
        channel_name(str): name of the neptune channel. Default is 'image_dir_snapshots'.
        experiment(neptune.experiemnts.Experiment or None): if the data should be logged to a particular
           neptune experiment it can be passed here. By default it is logged to the current experiment.
        seed(int): random state for the sampling of images.

    Examples:
        Initialize Neptune::

            import neptune
            from neptunecontrib.versioning.data import log_image_dir_snapshots
            neptune.init('USER_NAME/PROJECT_NAME')

        Log visual snapshot of image directory::

            PATH = 'train_dir/'
            with neptune.create_experiment():
                log_image_dir_snapshots(PATH)

    """
    _exp = experiment if experiment else neptune

    figs = _get_collated_images(image_dir, sample=sample, seed=seed)
    for fig in figs:
        send_figure(fig, channel_name=channel_name, experiment=_exp)


def _md5_hash_path(path):
    if os.path.isdir(path):
        return _md5_hash_dir(path)
    elif os.path.isfile(path):
        return _md5_hash_file(path)
    else:
        raise NotImplementedError


def _md5_hash_file(filepath):
    hash_md5 = hashlib.md5()
    hash_md5 = _update_hash_md5(hash_md5, filepath)
    return hash_md5.hexdigest()


def _md5_hash_dir(dirpath):
    hash_md5 = hashlib.md5()

    for root, _, files in os.walk(dirpath):
        for names in files:
            filepath = os.path.join(root, names)

            # Hash the path and add to the digest to account for empty files/directories
            hash_md5.update(hashlib.sha1(filepath[len(dirpath):].encode()).digest())

            if os.path.isfile(filepath):
                hash_md5 = _update_hash_md5(hash_md5, filepath)

    return hash_md5.hexdigest()


def _md5_hash_bucket(bucket_name, path):
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(bucket_name)

    hash_md5 = hashlib.md5()

    for obj in bucket.objects.all():
        if obj.key.startswith(path):
            hash_md5.update(obj.e_tag.encode('utf-8'))

    return hash_md5.hexdigest()


def _update_hash_md5(hash_md5, filepath):
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5


def _get_collated_images(image_dir, sample, seed):
    np.random.seed(seed)

    labels = _get_labels(image_dir)
    filepaths = _get_filepaths(image_dir)

    figures = []
    if labels:
        for label in labels:
            label_paths = [path for path in filepaths
                           if path.startswith(os.path.join(image_dir, label))]
            if len(label_paths) > sample:
                sample_paths = np.random.choice(label_paths, size=sample, replace=False)
            else:
                sample_paths = label_paths
            collated_image = _get_collated_image(sample_paths, label)
            figures.append(collated_image)
    else:
        if len(filepaths) > sample:
            sample_paths = np.random.choice(filepaths, size=sample, replace=False)
        else:
            sample_paths = filepaths
        collated_image = _get_collated_image(sample_paths)
        figures.append(collated_image)

    return figures


def _get_labels(dir_path):
    labels = []
    for fname in os.listdir(dir_path):
        if os.path.isdir(os.path.join(dir_path, fname)):
            labels.append(fname)
    return labels


def _get_filepaths(dir_path):
    filepaths = []
    for path, _, files in os.walk(dir_path):
        for name in files:
            filepaths.append(os.path.join(path, name))
    return filepaths


def _get_collated_image(filepaths, label=None, figsize=(16, 12), title_size=30):
    n = len(filepaths)
    yn, xn = int(np.floor(np.sqrt(n))), int(np.ceil(np.sqrt(n)))

    fig, axs = plt.subplots(yn, xn, figsize=figsize)
    fig.suptitle(label, fontsize=title_size)

    for i, filepath in enumerate(filepaths):
        yi, xi = i // xn, i % xn
        image = plt.imread(filepath)
        axs[yi, xi].imshow(image)
        axs[yi, xi].set_xticks([])
        axs[yi, xi].set_yticks([])
    plt.tight_layout()
    fig.subplots_adjust(top=0.92)

    return fig
