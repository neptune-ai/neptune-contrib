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
from functools import reduce

import matplotlib.pyplot as plt
import numpy as np
import neptune
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve, confusion_matrix, roc_auc_score

from neptunecontrib.monitoring.utils import fig2pil


def send_prediction_distribution(ctx, y_true, y_pred, figsize=(16,12)):
    """Converts matplotlib fig to PIL.Image

    Args:
        fig(`matplotlib.pyplot.figure`): Any matplotlib figure.

    Returns:
        `PIL.Image`: figure, converted to PIL Image.

    Examples:
        Create a figure:

        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> import seaborn as sns
        >>> fig = plt.figure(figsize=(16,12))
        >>> sns.distplot(np.random.random(100))

        Convert to PIL.

        >>> pil_figure = fig2pil(fig)

    Note:
        On some machines, using this function has cause matplotlib errors.
        What helped every time was to change matplotlib backend by adding the following snippet
        towards the top of your script:

        >>> import matplotlib
        >>> matplotlib.use('Agg')
    """
    fig_pred_dist = _plot_prediction_distribution(y_true, y_pred, figsize=figsize)
    npt_pred_dist = neptune.Image(name='chart', description='', 
                                  data=fig2pil(fig_pred_dist))
    ctx.channel_send('prediction_distribution', npt_pred_dist)
    
    
def send_roc_auc_curve(ctx, y_true, y_pred, figsize=(16,12)):
    """Converts matplotlib fig to PIL.Image

    Args:
        fig(`matplotlib.pyplot.figure`): Any matplotlib figure.

    Returns:
        `PIL.Image`: figure, converted to PIL Image.

    Examples:
        Create a figure:

        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> import seaborn as sns
        >>> fig = plt.figure(figsize=(16,12))
        >>> sns.distplot(np.random.random(100))

        Convert to PIL.

        >>> pil_figure = fig2pil(fig)

    Note:
        On some machines, using this function has cause matplotlib errors.
        What helped every time was to change matplotlib backend by adding the following snippet
        towards the top of your script:

        >>> import matplotlib
        >>> matplotlib.use('Agg')
    """
    fig_roc_auc = __plot_roc_auc_curve(y_true, y_pred, figsize=figsize)
    npt_roc_auc = neptune.Image(name='chart', description='', 
                                  data=fig2pil(fig_roc_auc))
    ctx.channel_send('roc_auc_curve', npt_roc_auc)
    
    
def send_confusion_matrix(ctx, y_true, y_pred, figsize=(16,12), threshold=0.5):
    """Converts matplotlib fig to PIL.Image

    Args:
        fig(`matplotlib.pyplot.figure`): Any matplotlib figure.

    Returns:
        `PIL.Image`: figure, converted to PIL Image.

    Examples:
        Create a figure:

        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> import seaborn as sns
        >>> fig = plt.figure(figsize=(16,12))
        >>> sns.distplot(np.random.random(100))

        Convert to PIL.

        >>> pil_figure = fig2pil(fig)

    Note:
        On some machines, using this function has cause matplotlib errors.
        What helped every time was to change matplotlib backend by adding the following snippet
        towards the top of your script:

        >>> import matplotlib
        >>> matplotlib.use('Agg')
    """
    fig_conf_matrix = _plot_confusion_matrix(y_true, y_pred>threshold, figsize=figsize)
    npt_conf_matrix = neptune.Image(name='chart', description='', 
                                  data=fig2pil(fig_conf_matrix))
    ctx.channel_send('confusion_matrix', npt_conf_matrix)


def _plot_prediction_distribution(y_true, y_pred, figsize=(16,12)):
    fig = plt.figure(figsize=figsize)
    df = pd.DataFrame({'prediction': y_pred,
                       'ground_truth': y_true
                       })
    
    sns.distplot(df[df['ground_truth'] == 0]['prediction'], label='negative')
    sns.distplot(df[df['ground_truth'] == 1]['prediction'], label='positive')

    plt.legend(prop={'size': 16}, title = 'Labels')
    return fig


def _plot_roc_auc_curve(y_true, y_pred,figsize=(16,12)):
    fig = plt.figure(figsize=figsize)
    fpr, tpr, threshold = roc_curve(y_true, y_pred)
    score = roc_auc_score(y_true, y_pred)

    plt.plot(fpr, tpr, color='darkorange', label="ROC curve {}".format(score))
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.title("Results")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    
    return fig


def _plot_confusion_matrix(y_true, y_pred, figsize=(16,12)):
    cm = confusion_matrix(y_true, y_pred)
    fig = plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, cmap="YlGnBu", fmt="d")
    return fig
