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

from itertools import product
import warnings

import matplotlib.pyplot as plt
import neptune

from neptunecontrib.api import pickle_and_log_artifact


def axes2fig(axes, fig=None):
    """Converts ndarray of matplotlib object to matplotlib figure.

    Scikit-optimize plotting functions return ndarray of axes. This can be tricky
    to work with so you can use this function to convert it to the standard figure format.

    Args:
        axes(`numpy.ndarray`): Array of matplotlib axes objects.
        fig('matplotlib.figure.Figure'): Matplotlib figure on which you may want to plot
            your axes. Default None.

    Returns:
        'matplotlib.figure.Figure': Matplotlib figure with axes objects as subplots.

    Examples:
        Assuming you have a `scipy.optimize.OptimizeResult` object you want to plot::

            from skopt.plots import plot_evaluations
            eval_plot = plot_evaluations(result, bins=20)
            >>> type(eval_plot)
                numpy.ndarray

            from neptunecontrib.viz.utils import axes2fig
            fig = axes2fig(eval_plot)
            >>> fig
                matplotlib.figure.Figure

    """
    try:
        h, w = axes.shape
        if not fig:
            fig = plt.figure(figsize=(h * 3, w * 3))
        for i, j in product(range(h), range(w)):
            fig._axstack.add(fig._make_key(axes[i, j]), axes[i, j])
    except AttributeError:
        if not fig:
            fig = plt.figure(figsize=(6, 6))
        fig._axstack.add(fig._make_key(axes), axes)
    return fig


def send_figure(fig, channel_name='figures', experiment=None):
    message = """neptunecontrib.monitoring.utils send_figure functionality is now available in neptune-client.
    You should simply use neptune.log_image('channel_name', fig) where you used send_figure('channel_name', fig) before.
    send_figure will be removed in future releases.
    """
    warnings.warn(message)

    _exp = experiment if experiment else neptune
    _exp.log_image(channel_name, fig)


def pickle_and_send_artifact(obj, filename, experiment=None):
    message = """neptunecontrib.monitoring.utils pickle_and_send_artifact was moved to neptunecontrib.api
    and renamed to pickle_and_log_artifact. You should use ``from neptunecontrib.api import pickle_and_log_artifact``
    neptunecontrib.logging.log_chart will be removed in future releases.
    """
    warnings.warn(message)

    pickle_and_log_artifact(obj, filename, experiment)
