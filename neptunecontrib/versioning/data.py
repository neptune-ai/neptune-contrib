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

import neptune


def log_data_version(filepath, prefix='', experiment=None):
    """Logs data version to Neptune

    For a path it calculates the hash and logs it along with the path itself as a property to Neptune experiment.
    Path to dataset, which can be a file or directory.

    Args:
        filepath(str): path to the file or directory,
        prefix(str): Prefix that will be added before 'ata_version' and 'data_path'
        experiment(neptune.experiemnts.Experiment or None): if the data should be logged to a particular
           neptune experiment it can be passed here. By default it is logged to the current experiment.

    Examples:
        Initialize Neptune

         >>> import neptune
         >>> from neptunecontrib.versioning.data import log_data_version
         >>> neptune.init('USER_NAME/PROJECT_NAME')

         Log data from filepath

         >>> FILEPATH = '/path/to/data/my_data.csv'
         >>> with neptune.create_experiment():
         >>>    log_data_version(FILEPATH)

    """

    _exp = experiment if experiment else neptune

    _exp.set_property('{}data_path'.format(prefix), filepath)
    _exp.set_property('{}data_version'.format(prefix), _md5_hash_path(filepath))


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


def _update_hash_md5(hash_md5, filepath):
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5
