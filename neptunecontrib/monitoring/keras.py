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

import neptune
from neptune.exceptions import LibraryNotInstalled, NeptuneException

# Note: we purposefully try to import `tensorflow.keras.callbacks.Callback`
# before `keras.callbacks.Callback` because the former is compatible with both
# `tensorflow.keras` and `keras`, while the latter is only compatible
# with `keras`. See https://github.com/keras-team/keras/issues/14125
try:
    from tensorflow.keras.callbacks import Callback
except ImportError:
    try:
        from keras.callbacks import Callback
    except ImportError:
        raise LibraryNotInstalled('Keras')


class NeptuneMonitor(Callback):
    """Logs Keras metrics to Neptune.

    Goes over the `last_metrics` and `smooth_loss` after each batch and epoch
    and logs them to appropriate Neptune channels.

    See the example experiment here TODO

    Args:
        experiment: `neptune.Experiment`, optional:
            Neptune experiment. If not provided, falls back on the current
            experiment.
        prefix: str, optional:
            Prefix that should be added before the `metric_name`
            and `valid_name` before logging to the appropriate channel.
            Defaul is empty string ('').

    Examples:

        Now, create Neptune experiment, instantiate the monitor and pass
        it to callbacks::

            TODO update for keras

    Note:
        You need to have Keras or Tensorflow 2 installed on your computer to use this module.
    """

    def __init__(self, experiment=None, prefix=''):
        super().__init__()
        self._exp = experiment if experiment else neptune
        self._prefix = prefix

    def _log_metrics(self, logs, trigger):
        if not logs:
            return

        prefix = self._prefix + trigger + '_'

        for metric, value in logs.items():
            try:
                if metric in ('batch', 'size'):
                    continue
                name = prefix + metric
                self._exp.log_metric(name, value, y=None)
            except NeptuneException:
                pass

    def on_batch_end(self, batch, logs=None):  # pylint:disable=unused-argument
        self._log_metrics(logs, 'batch')

    def on_epoch_end(self, epoch, logs=None):  # pylint:disable=unused-argument
        self._log_metrics(logs, 'epoch')
