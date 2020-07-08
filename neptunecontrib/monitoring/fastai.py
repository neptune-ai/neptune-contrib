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

import sys

import neptune

if sys.version_info[0] == 3 and sys.version_info[1] >= 6:
    from fastai.basic_train import LearnerCallback
else:
    class LearnerCallback:
        pass


    LearnerCallback.__module__ = 'fastai.basic_train'


class NeptuneMonitor(LearnerCallback):
    """Logs metrics from the fastai learner to Neptune.

    Goes over the `last_metrics` and `smooth_loss` after each batch and epoch
    and logs them to appropriate Neptune channels.

    See the example experiment here
    https://ui.neptune.ai/neptune-ai/neptune-examples/e/NEP-493/charts.


    Args:
        experiment(`neptune.experiments.Experiment`): Neptune experiment.
        prefix(str): Prefix that should be added before the `metric_name`
            and `valid_name` before logging to the appropriate channel.
            Defaul is ''.

    Examples:
        Prepare data::

            from fastai.vision import *
            path = untar_data(URLs.MNIST_TINY)

            data = ImageDataBunch.from_folder(path, ds_tfms=(rand_pad(2, 28), []), bs=64)
            data.normalize(imagenet_stats)

            learn = cnn_learner(data, models.resnet18, metrics=accuracy)

            learn.lr_find()
            learn.recorder.plot()

        Now, create Neptune experiment, instantiate the monitor and pass
        it to callbacks::

            import neptune
            from neptunecontrib.monitoring.fastai import NeptuneMonitor

            neptune.init(qualified_project_name='USER_NAME/PROJECT_NAME')

            with neptune.create_experiment():
                learn = create_cnn(data, models.resnet18,
                                   metrics=accuracy,
                                   callbacks_fns=[NeptuneMonitor])
                learn.fit_one_cycle(20, 1e-2)

    Note:
        you need to have the fastai library installed on your computer to use this module.
    """

    def __init__(self, learn=None, experiment=None, prefix=''):
        self._exp = experiment if experiment else neptune
        self._prefix = prefix
        if learn is not None:
            super().__init__(learn)

    def on_epoch_end(self, **kwargs):
        self._exp.send_metric(self._prefix + 'train_smooth_loss', float(kwargs['smooth_loss']))
        metric_values = kwargs['last_metrics']
        metric_names = ['valid_last_loss'] + kwargs['metrics']
        for metric_value, metric_name in zip(metric_values, metric_names):
            if metric_value is None:
                continue

            metric_name = getattr(metric_name, '__name__', metric_name)
            self._exp.send_metric(self._prefix + str(metric_name), float(metric_value))

    def on_batch_end(self, last_loss, iteration, train, **kwargs):
        if iteration == 0 or not train:
            return
        self._exp.send_metric('{}last_loss'.format(self._prefix), last_loss)
