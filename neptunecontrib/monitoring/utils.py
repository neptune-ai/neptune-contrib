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
import os
import tempfile

import joblib
import neptune
import matplotlib.pyplot as plt


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
    """Logs matplotlib figure to Neptune.

    Logs any figure from matplotlib to specified image channel.
    By default it logs to 'figures' and you can log multiple images to the same channel.

    Args:
        channel_name(str): name of the neptune channel. Default is 'figures'.
        experiment(`neptune.experiments.Experiment`): Neptune experiment. Default is None.
        fig(`matplotlib.figure`): Matplotlib figure object

    Examples:
        Initialize Neptune::

            import neptune
            neptune.init('USER_NAME/PROJECT_NAME')

        Create random data:::

            import numpy as np
            table = np.random.random((10,10))

        Plot and log to Neptune::

            import matplotlib.pyplot as plt
            import seaborn as sns
            from neptunecontrib.monitoring.utils import send_figure

            with neptune.create_experiment():
                fig, ax = plt.subplots()
                sns.heatmap(table,ax=ax)
                send_figure(fig)

    """
    _exp = experiment if experiment else neptune

    with tempfile.NamedTemporaryFile(suffix='.png') as f:
        fig.savefig(f.name)
        _exp.send_image(channel_name, f.name)


def pickle_and_send_artifact(obj, filename, experiment=None):
    """Logs picklable object to Neptune.

    Pickles and logs your object to Neptune under specified filename.

    Args:
        obj: Picklable object.
        filename(str): filename under which object will be saved.
        experiment(`neptune.experiments.Experiment`): Neptune experiment. Default is None.

    Examples:
        Initialize Neptune::

            import neptune
            neptune.init('USER_NAME/PROJECT_NAME')

        Create RandomForest object and log to Neptune::

            from sklearn.ensemble import RandomForestClassifier
            from neptunecontrib.monitoring.utils import pickle_and_send_artifact

            with neptune.create_experiment():
                rf = RandomForestClassifier()
                pickle_and_send_artifact(rf, 'rf')
    """
    _exp = experiment if experiment else neptune

    with tempfile.TemporaryDirectory() as d:
        filename = os.path.join(d, filename)
        joblib.dump(obj, filename)
        _exp.send_artifact(filename)
