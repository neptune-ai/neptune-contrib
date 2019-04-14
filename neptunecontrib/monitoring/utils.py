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
        Assuming you have a `scipy.optimize.OptimizeResult` object you want to plot.

        >>> from skopt.plots import plot_evaluations
        >>> eval_plot = plot_evaluations(result, bins=20)
        >>> type(eval_plot)
        numpy.ndarray

        >>> from neptunecontrib.viz.utils import axes2fig
        >>> fig = axes2fig(eval_plot)
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
