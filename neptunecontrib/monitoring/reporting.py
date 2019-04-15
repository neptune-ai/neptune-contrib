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

import tempfile

import matplotlib.pyplot as plt
import neptune
import pandas as pd
import seaborn as sns
from scikitplot.metrics import plot_roc, plot_precision_recall, plot_confusion_matrix


def send_binary_classification_report(y_true, y_pred,
                                      experiment=None,
                                      threshold=0.5,
                                      figsize=(16, 12),
                                      channel_name='classification report'):
    """Creates binary classification report and logs it in Neptune.

    This function creates ROC AUC curve, confusion matrix, precision recall curve and
    prediction distribution charts and logs it to the 'classification report' channel in Neptune.

    Args:
        y_true (array-like, shape (n_samples)): Ground truth (correct) target values.
        y_pred (array-like, shape (n_samples, 2)): Predictions both for negative and positive class
            in the float format.
        experiment(`neptune.experiments.Experiment`): Neptune experiment. Default is None.
        threshold(float): threshold to be applied for the class asignment.
        figsize(tuple): size of the matplotlib.pyplot figure object
        channel_name(str): name of the neptune channel. Default is 'classification report'.

    Examples:
        Train the model and make predictions on test.

        >>> from sklearn.datasets import make_classification
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> from sklearn.model_selection import train_test_split
        >>> from sklearn.metrics import classification_report
        >>>
        >>> X, y = make_classification(n_samples=2000)
        >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        >>>
        >>> model = RandomForestClassifier()
        >>> model.fit(X_train, y_train)
        >>>
        >>> y_test_pred = model.predict_proba(X_test)

        Log classification report to Neptune.

        >>> import neptune
        >>> from neptunecontrib.monitoring.reporting import send_binary_classification_report
        >>>
        >>> neptune.init(qualified_project_name='USER_NAME/PROJECT_NAME')
        >>> with neptune.create_experiment():
        >>>    send_binary_classification_report(y_test, y_test_pred)

    """

    _exp = experiment if experiment else neptune

    fig = plot_binary_classification_report(y_true, y_pred, threshold=threshold, figsize=figsize)
    with tempfile.NamedTemporaryFile(suffix='.png') as f:
        fig.savefig(f.name)
        _exp.send_image(channel_name, f.name)


def send_prediction_distribution(y_true, y_pred,
                                 experiment=None,
                                 figsize=(16, 12),
                                 channel_name='prediction distribution'):
    """Creates prediction distribution chart and logs it in Neptune.

    Args:
        y_true (array-like, shape (n_samples)): Ground truth (correct) target values.
        y_pred (array-like, shape (n_samples)): Predictions for the positive class in the float format.
        experiment(`neptune.experiments.Experiment`): Neptune experiment. Default is None.
        figsize(tuple): size of the matplotlib.pyplot figure object
        channel_name(str): name of the neptune channel. Default is 'prediction distribution'.

    Examples:
        Train the model and make predictions on test.

        >>> from sklearn.datasets import make_classification
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> from sklearn.model_selection import train_test_split
        >>> from sklearn.metrics import classification_report
        >>>
        >>> X, y = make_classification(n_samples=2000)
        >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        >>>
        >>> model = RandomForestClassifier()
        >>> model.fit(X_train, y_train)
        >>>
        >>> y_test_pred = model.predict_proba(X_test)

        Log prediction distribution to Neptune.

        >>> import neptune
        >>> from neptunecontrib.monitoring.reporting import send_prediction_distribution
        >>>
        >>> neptune.init(qualified_project_name='USER_NAME/PROJECT_NAME')
        >>>
        >>> with neptune.create_experiment():
        >>>    send_prediction_distribution(ctx, y_test, y_test_pred[:, 1])

    """

    _exp = experiment if experiment else neptune

    fig, ax = plt.subplots(figsize=figsize)
    plot_prediction_distribution(y_true, y_pred, ax=ax)

    with tempfile.NamedTemporaryFile(suffix='.png') as f:
        fig.savefig(f.name)
        _exp.send_image(channel_name, f.name)


def send_roc_auc_curve(y_true, y_pred,
                       experiment=None,
                       figsize=(16, 12),
                       channel_name='ROC AUC curve'):
    """Creates ROC AUC curve and logs it in Neptune.

    Args:
        y_true (array-like, shape (n_samples)): Ground truth (correct) target values.
        y_pred (array-like, shape (n_samples, 2)): Predictions both for negative and positive class
            in the float format.
        experiment(`neptune.experiments.Experiment`): Neptune experiment. Default is None.
        figsize(tuple): size of the matplotlib.pyplot figure object
        channel_name(str): name of the neptune channel. Default is 'ROC AUC curve'.

    Examples:
        Train the model and make predictions on test.

        >>> from sklearn.datasets import make_classification
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> from sklearn.model_selection import train_test_split
        >>> from sklearn.metrics import classification_report
        >>>
        >>> X, y = make_classification(n_samples=2000)
        >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        >>>
        >>> model = RandomForestClassifier()
        >>> model.fit(X_train, y_train)
        >>>
        >>> y_test_pred = model.predict_proba(X_test)

        Log classification report to Neptune.

        >>> import neptune
        >>> from neptunecontrib.monitoring.reporting import send_roc_auc_curve
        >>>
        >>> neptune.init(qualified_project_name='USER_NAME/PROJECT_NAME')
        >>>
        >>> with neptune.create_experiment():
        >>>    send_roc_auc_curve(ctx, y_test, y_test_pred)

    """

    _exp = experiment if experiment else neptune

    fig, ax = plt.subplots(figsize=figsize)
    plot_roc(y_true, y_pred, ax=ax)

    with tempfile.NamedTemporaryFile(suffix='.png') as f:
        fig.savefig(f.name)
        _exp.send_image(channel_name, f.name)


def send_confusion_matrix(y_true, y_pred,
                          experiment=None,
                          figsize=(16, 12),
                          channel_name='confusion_matrix'):
    """Creates ROC AUC curve and logs it in Neptune.

    Args:
        y_true (array-like, shape (n_samples)): Ground truth (correct) target values.
        y_pred (array-like, shape (n_samples)): Positive class predictions in the binary format.
        experiment(`neptune.experiments.Experiment`): Neptune experiment. Default is None.
        figsize(tuple): size of the matplotlib.pyplot figure object
        channel_name(str): name of the neptune channel. Default is 'ROC AUC curve'.

    Examples:
        Train the model and make predictions on test.

        >>> from sklearn.datasets import make_classification
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> from sklearn.model_selection import train_test_split
        >>> from sklearn.metrics import classification_report
        >>>
        >>> X, y = make_classification(n_samples=2000)
        >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        >>>
        >>> model = RandomForestClassifier()
        >>> model.fit(X_train, y_train)
        >>>
        >>> y_test_pred = model.predict_proba(X_test)

        Log classification report to Neptune.

        >>> import neptune
        >>> from neptunecontrib.monitoring.reporting import send_confusion_matrix
        >>>
        >>> neptune.init(qualified_project_name='USER_NAME/PROJECT_NAME')
        >>>
        >>> with neptune.create_experiment():
        >>>    send_confusion_matrix(ctx, y_test, y_test_pred[:, 1] > 0.5)

    """
    fig, ax = plt.subplots(figsize=figsize)
    plot_confusion_matrix(y_true, y_pred, ax=ax)

    _exp = experiment if experiment else neptune

    with tempfile.NamedTemporaryFile(suffix='.png') as f:
        fig.savefig(f.name)
        _exp.send_image(channel_name, f.name)


def send_precision_recall(y_true, y_pred,
                          experiment=None,
                          figsize=(16, 12),
                          channel_name='precision_recall_curve'):
    """Creates precision recall curve and logs it in Neptune.

    Args:
        ctx(`neptune.Context`): Neptune context.
        y_true (array-like, shape (n_samples)): Ground truth (correct) target values.
        y_pred (array-like, shape (n_samples, 2)): Predictions both for negative and positive class
            in the float format.
        experiment(`neptune.experiments.Experiment`): Neptune experiment. Default is None.
        figsize(tuple): size of the matplotlib.pyplot figure object
        channel_name(str): name of the neptune channel. Default is 'ROC AUC curve'.

    Examples:
        Train the model and make predictions on test.

        >>> from sklearn.datasets import make_classification
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> from sklearn.model_selection import train_test_split
        >>> from sklearn.metrics import classification_report
        >>>
        >>> X, y = make_classification(n_samples=2000)
        >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        >>>
        >>> model = RandomForestClassifier()
        >>> model.fit(X_train, y_train)
        >>>
        >>> y_test_pred = model.predict_proba(X_test)

        Log classification report to Neptune.

        >>> import neptune
        >>> from neptunecontrib.monitoring.reporting import send_precision_recall
        >>>
        >>> neptune.init(qualified_project_name='USER_NAME/PROJECT_NAME')
        >>>
        >>> with neptune.create_experiment():
        >>>    send_precision_recall(ctx, y_test, y_test_pred)

    """

    _exp = experiment if experiment else neptune

    fig, ax = plt.subplots(figsize=figsize)
    plot_precision_recall(y_true, y_pred, ax=ax)

    with tempfile.NamedTemporaryFile(suffix='.png') as f:
        fig.savefig(f.name)
        _exp.send_image(channel_name, f.name)


def plot_binary_classification_report(y_true, y_pred, threshold=0.5, figsize=(16, 12)):
    """Creates binary classification report.

    This function creates ROC AUC curve, confusion matrix, precision recall curve and
    prediction distribution charts and logs it to the 'classification report' channel in Neptune.

    Args:
        y_true (array-like, shape (n_samples)): Ground truth (correct) target values.
        y_pred (array-like, shape (n_samples, 2)): Predictions both for negative and positive class
            in the float format.
        threshold(float): threshold to be applied for the class asignment.
        figsize(tuple): size of the matplotlib.pyplot figure object

    Returns:
         (`matplotlib.figure`): Figure object with binary classification report.

    Examples:
        Train the model and make predictions on test.

        >>> from sklearn.datasets import make_classification
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> from sklearn.model_selection import train_test_split
        >>> from sklearn.metrics import classification_report
        >>>
        >>> X, y = make_classification(n_samples=2000)
        >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        >>>
        >>> model = RandomForestClassifier()
        >>> model.fit(X_train, y_train)
        >>>
        >>> y_test_pred = model.predict_proba(X_test)

        Plot binary classification report.

        >>> from neptunecontrib.monitoring.reporting import plot_binary_classification_report
        >>>
        >>> plot_binary_classification_report(y_test, y_test_pred)

    """
    fig, axs = plt.subplots(2, 2, figsize=figsize)
    plot_roc(y_true, y_pred, ax=axs[0, 0])
    plot_precision_recall(y_true, y_pred, ax=axs[0, 1])
    plot_prediction_distribution(y_true, y_pred[:, 1], ax=axs[1, 0])
    plot_confusion_matrix(y_true, y_pred[:, 1] > threshold, ax=axs[1, 1])
    fig.tight_layout()
    return fig


def plot_prediction_distribution(y_true, y_pred, ax=None, figsize=None):
    """Generates prediction distribution plot from predictions and true labels.

    Args:
        y_true (array-like, shape (n_samples)):
            Ground truth (correct) target values.
        y_pred (array-like, shape (n_samples)):
            Estimated targets as returned by a classifier.
        ax (:class:`matplotlib.axes.Axes`, optional): The axes upon which to
            plot the curve. If None, the plot is drawn on a new set of axes.
        figsize (2-tuple, optional): Tuple denoting figure size of the plot
            e.g. (6, 6). Defaults to ``None``.

    Returns:
        ax (:class:`matplotlib.axes.Axes`): The axes on which the plot was
            drawn.

    Examples:
        Train the model and make predictions on test.

        >>> from sklearn.datasets import make_classification
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> from sklearn.model_selection import train_test_split
        >>> from sklearn.metrics import classification_report
        >>>
        >>> X, y = make_classification(n_samples=2000)
        >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        >>>
        >>> model = RandomForestClassifier()
        >>> model.fit(X_train, y_train)
        >>>
        >>> y_test_pred = model.predict_proba(X_test)

        Plot prediction distribution.

        >>> from neptunecontrib.monitoring.reporting import plot_prediction_distribution
        >>>
        >>> plot_prediction_distribution(y_test, y_test_pred[:, 1])
    """

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=figsize)

    ax.set_title('Prediction Distribution', fontsize='large')

    df = pd.DataFrame({'Prediction': y_pred,
                       'True label': y_true})

    sns.distplot(df[df['True label'] == 0]['Prediction'], label='negative', ax=ax)
    sns.distplot(df[df['True label'] == 1]['Prediction'], label='positive', ax=ax)

    ax.legend(prop={'size': 16}, title='Labels')

    return ax
