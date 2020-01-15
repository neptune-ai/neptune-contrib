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
import matplotlib.pyplot as plt
import neptune
from neptunecontrib.monitoring.utils import send_figure
import numpy as np
import pandas as pd
import scikitplot.metrics as plt_metrics
from scikitplot.helpers import binary_ks_curve
import seaborn as sns
import sklearn.metrics as sk_metrics


def log_binary_classification_metrics(y_true, y_pred, threshold=0.5, experiment=None, prefix=''):
    """Creates metric charts and calculates classification metrics and logs them to Neptune.

    Class-based metrics that are logged: 'accuracy', 'precision', 'recall', 'f1_score', 'f2_score',
    'matthews_corrcoef', 'cohen_kappa', 'true_positive_rate', 'true_negative_rate', 'positive_predictive_value',
    'negative_predictive_value', 'false_positive_rate', 'false_negative_rate', 'false_discovery_rate'
    For each class-based metric, a curve with metric/threshold is logged to 'metrics_by_threshold' channel.

    Losses that are logged: 'brier_loss', 'log_loss'

    Other metrics that are logged: 'roc_auc', 'ks_statistic', 'avg_precision'

    Curves that are logged: 'roc_auc', 'precision_recall_curve', 'ks_statistic_curve', 'cumulative_gain_curve',
    'lift_curve',

    Args:
        y_true (array-like, shape (n_samples)): Ground truth (correct) target values.
        y_pred (array-like, shape (n_samples, 2)): Predictions for classes 0 and 1 with values from 0 to 1.
        experiment(`neptune.experiments.Experiment`): Neptune experiment. Default is None.
        threshold (float): Threshold that calculates a class for class-based metrics. Default is 0.5.
        prefix(str): Prefix that will be added before metric name when logged to Neptune.

    Examples:
        Train the model and make predictions on test::

            from sklearn.datasets import make_classification
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import classification_report

            X, y = make_classification(n_samples=2000)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

            model = RandomForestClassifier()
            model.fit(X_train, y_train)

            y_test_pred = model.predict_proba(X_test)

        Log metrics and performance curves to Neptune::

            import neptune
            from neptunecontrib.monitoring.metrics import log_binary_classification_metrics

            neptune.init()
            with neptune.create_experiment():
                log_binary_classification_metrics(y_test, y_test_pred, threshold=0.5)

        Check out this experiment https://ui.neptune.ai/o/neptune-ai/org/binary-classification-metrics/e/BIN-101/logs.

    """
    assert len(y_pred.shape) == 2, 'y_pred needs to be (n_samples, 2), use expand_prediction helper to format it'

    _exp = experiment if experiment else neptune

    log_confusion_matrix(y_true, y_pred[:, 1] > threshold, experiment=_exp, prefix=prefix)
    log_classification_report(y_true, y_pred[:, 1] > threshold, experiment=_exp, prefix=prefix)
    log_class_metrics(y_true, y_pred[:, 1] > threshold, experiment=_exp, prefix=prefix)
    log_class_metrics_by_threshold(y_true, y_pred[:, 1], experiment=_exp, prefix=prefix)
    log_roc_auc(y_true, y_pred, experiment=_exp, prefix=prefix)
    log_precision_recall_auc(y_true, y_pred, experiment=_exp, prefix=prefix)
    log_brier_loss(y_true, y_pred[:, 1], experiment=_exp, prefix=prefix)
    log_log_loss(y_true, y_pred, experiment=_exp, prefix=prefix)
    log_ks_statistic(y_true, y_pred, experiment=_exp, prefix=prefix)
    log_cumulative_gain(y_true, y_pred, experiment=_exp, prefix=prefix)
    log_lift_curve(y_true, y_pred, experiment=_exp, prefix=prefix)
    log_prediction_distribution(y_true, y_pred[:, 1], experiment=_exp, prefix=prefix)


def log_confusion_matrix(y_true, y_pred_class, experiment=None, channel_name='metric_charts', prefix=''):
    """Creates a confusion matrix figure and logs it in Neptune.

    Args:
        y_true (array-like, shape (n_samples)): Ground truth (correct) target values.
        y_pred_class (array-like, shape (n_samples)): Class predictions with values 0 or 1.
        experiment(`neptune.experiments.Experiment`): Neptune experiment. Default is None.
        channel_name(str): name of the neptune channel. Default is 'metric_charts'.
        prefix(str): Prefix that will be added before metric name when logged to Neptune.

    Examples:
        Train the model and make predictions on test::

            from sklearn.datasets import make_classification
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import classification_report

            X, y = make_classification(n_samples=2000)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

            model = RandomForestClassifier()
            model.fit(X_train, y_train)

            y_test_pred = model.predict_proba(X_test)

        Log confusion matrix to Neptune::

            import neptune
            from neptunecontrib.monitoring.metrics import log_confusion_matrix

            neptune.init()
            with neptune.create_experiment():
                log_confusion_matrix(y_test, y_test_pred[:,1]>0.5)

        Check out this experiment https://ui.neptune.ai/o/neptune-ai/org/binary-classification-metrics/e/BIN-101/logs.

    """
    assert len(y_pred_class.shape) == 1, 'y_pred_class needs to be 1D class prediction with values 0, 1'

    _exp = experiment if experiment else neptune

    fig, ax = plt.subplots()
    _plot_confusion_matrix(y_true, y_pred_class, ax=ax)
    send_figure(fig, channel_name=prefix + channel_name, experiment=_exp)
    plt.close()


def log_classification_report(y_true, y_pred_class, experiment=None, channel_name='metric_charts', prefix=''):
    """Creates a figure with classifiction report table and logs it in Neptune.

    Args:
        y_true (array-like, shape (n_samples)): Ground truth (correct) target values.
        y_pred_class (array-like, shape (n_samples)): Class predictions with values 0 or 1.
        experiment(`neptune.experiments.Experiment`): Neptune experiment. Default is None.
        channel_name(str): name of the neptune channel. Default is 'metric_charts'.
        prefix(str): Prefix that will be added before metric name when logged to Neptune.

    Examples:
        Train the model and make predictions on test::

            from sklearn.datasets import make_classification
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import classification_report

            X, y = make_classification(n_samples=2000)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

            model = RandomForestClassifier()
            model.fit(X_train, y_train)

            y_test_pred = model.predict_proba(X_test)

        Log classification report to Neptune::

            import neptune
            from neptunecontrib.monitoring.metrics import log_classification_report

            neptune.init()
            with neptune.create_experiment():
                log_classification_report(y_test, y_test_pred[:,1]>0.5)

        Check out this experiment https://ui.neptune.ai/o/neptune-ai/org/binary-classification-metrics/e/BIN-101/logs.

    """
    assert len(y_pred_class.shape) == 1, 'y_pred_class needs to be 1D class prediction with values 0, 1'

    _exp = experiment if experiment else neptune

    fig = _plot_classification_report(y_true, y_pred_class)
    send_figure(fig, channel_name=prefix + channel_name, experiment=_exp)
    plt.close()


def log_class_metrics(y_true, y_pred_class, experiment=None, prefix=''):
    """Calculates and logs all class-based metrics to Neptune.

    Metrics that are logged: 'accuracy', 'precision', 'recall', 'f1_score', 'f2_score', 'matthews_corrcoef',
    'cohen_kappa', 'true_positive_rate', 'true_negative_rate', 'positive_predictive_value',
    'negative_predictive_value', 'false_positive_rate', 'false_negative_rate', 'false_discovery_rate'

    Args:
        y_true (array-like, shape (n_samples)): Ground truth (correct) target values.
        y_pred_class (array-like, shape (n_samples)): Class predictions with values 0 or 1.
        experiment(`neptune.experiments.Experiment`): Neptune experiment. Default is None.
        prefix(str): Prefix that will be added before metric name when logged to Neptune.

    Examples:
        Train the model and make predictions on test::

            from sklearn.datasets import make_classification
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import classification_report

            X, y = make_classification(n_samples=2000)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

            model = RandomForestClassifier()
            model.fit(X_train, y_train)

            y_test_pred = model.predict_proba(X_test)

        Log class metrics to Neptune::

            import neptune
            from neptunecontrib.monitoring.metrics import log_class_metrics

            neptune.init()
            with neptune.create_experiment():
                log_class_metrics(y_test, y_test_pred[:,1]>0.5)

        Check out this experiment https://ui.neptune.ai/o/neptune-ai/org/binary-classification-metrics/e/BIN-101/logs.

    """
    assert len(y_pred_class.shape) == 1, 'y_pred_class needs to be 1D class prediction with values 0, 1'

    _exp = experiment if experiment else neptune

    scores = _class_metrics(y_true, y_pred_class)
    for metric_name, score in scores.items():
        _exp.log_metric(prefix + metric_name, score)


def log_class_metrics_by_threshold(y_true, y_pred_pos, experiment=None, channel_name='metrics_by_threshold', prefix=''):
    """Creates metric/threshold charts for each metric and logs them to Neptune.

    Metrics for which charsta re created and logged are: 'accuracy', 'precision', 'recall', 'f1_score', 'f2_score',
    'matthews_corrcoef', 'cohen_kappa', 'true_positive_rate', 'true_negative_rate', 'positive_predictive_value',
    'negative_predictive_value', 'false_positive_rate', 'false_negative_rate', 'false_discovery_rate'

    Args:
        y_true (array-like, shape (n_samples)): Ground truth (correct) target values.
        y_pred_pos (array-like, shape (n_samples)): Score predictions with values from 0 to 1.
        experiment(`neptune.experiments.Experiment`): Neptune experiment. Default is None.
        channel_name(str): name of the neptune channel. Default is 'metrics_by_threshold'.
        prefix(str): Prefix that will be added before metric name when logged to Neptune.

    Examples:
        Train the model and make predictions on test::

            from sklearn.datasets import make_classification
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import classification_report

            X, y = make_classification(n_samples=2000)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

            model = RandomForestClassifier()
            model.fit(X_train, y_train)

            y_test_pred = model.predict_proba(X_test)

        Logs metric/threshold charts to Neptune::

            import neptune
            from neptunecontrib.monitoring.metrics import log_class_metrics_by_threshold

            neptune.init()
            with neptune.create_experiment():
                log_class_metrics_by_threshold(y_test, y_test_pred[:,1])

        Check out this experiment https://ui.neptune.ai/o/neptune-ai/org/binary-classification-metrics/e/BIN-101/logs.

    """
    assert len(y_pred_pos.shape) == 1, 'y_pred_pos needs to be 1D prediction for positive class'

    _exp = experiment if experiment else neptune

    figs = _plot_class_metrics_by_threshold(y_true, y_pred_pos)

    for fig in figs:
        send_figure(fig, channel_name=prefix + channel_name, experiment=_exp)
        plt.close()


def log_roc_auc(y_true, y_pred, experiment=None, channel_name='metric_charts', prefix=''):
    """Creates and logs ROC AUC curve and ROCAUC score to Neptune.

    Args:
        y_true (array-like, shape (n_samples)): Ground truth (correct) target values.
        y_pred (array-like, shape (n_samples, 2)): Predictions for classes 0 and 1 with values from 0 to 1.
        experiment(`neptune.experiments.Experiment`): Neptune experiment. Default is None.
        channel_name(str): name of the neptune channel. Default is 'metric_charts'.
        prefix(str): Prefix that will be added before metric name when logged to Neptune.

    Examples:
        Train the model and make predictions on test::

            from sklearn.datasets import make_classification
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import classification_report

            X, y = make_classification(n_samples=2000)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

            model = RandomForestClassifier()
            model.fit(X_train, y_train)

            y_test_pred = model.predict_proba(X_test)

        Logs ROCAUC curve and ROCAUC score to Neptune::

            import neptune
            from neptunecontrib.monitoring.metrics import log_roc_auc

            neptune.init()
            with neptune.create_experiment():
                log_roc_auc(y_test, y_test_pred)

        Check out this experiment https://ui.neptune.ai/o/neptune-ai/org/binary-classification-metrics/e/BIN-101/logs.

    """
    assert len(y_pred.shape) == 2, 'y_pred needs to be (n_samples, 2), use expand_prediction helper to format it'

    _exp = experiment if experiment else neptune

    roc_auc = sk_metrics.roc_auc_score(y_true, y_pred[:, 1])
    _exp.log_metric(prefix + 'roc_auc', roc_auc)

    fig, ax = plt.subplots()
    plt_metrics.plot_roc(y_true, y_pred, ax=ax)
    send_figure(fig, channel_name=prefix + channel_name, experiment=_exp)
    plt.close()


def log_precision_recall_auc(y_true, y_pred, experiment=None, channel_name='metric_charts', prefix=''):
    """Creates and logs Precision Recall curve and Average precision score to Neptune.

    Args:
        y_true (array-like, shape (n_samples)): Ground truth (correct) target values.
        y_pred (array-like, shape (n_samples, 2)): Predictions for classes 0 and 1 with values from 0 to 1.
        experiment(`neptune.experiments.Experiment`): Neptune experiment. Default is None.
        channel_name(str): name of the neptune channel. Default is 'metric_charts'.
        prefix(str): Prefix that will be added before metric name when logged to Neptune.

    Examples:
        Train the model and make predictions on test::

            from sklearn.datasets import make_classification
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import classification_report

            X, y = make_classification(n_samples=2000)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

            model = RandomForestClassifier()
            model.fit(X_train, y_train)

            y_test_pred = model.predict_proba(X_test)

        Logs Precision Recall curve and Average precision score to Neptune::

            import neptune
            from neptunecontrib.monitoring.metrics import log_precision_recall_auc

            neptune.init()
            with neptune.create_experiment():
                log_precision_recall_auc(y_test, y_test_pred)

        Check out this experiment https://ui.neptune.ai/o/neptune-ai/org/binary-classification-metrics/e/BIN-101/logs.

    """
    assert len(y_pred.shape) == 2, 'y_pred needs to be (n_samples, 2), use expand_prediction helper to format it'

    _exp = experiment if experiment else neptune

    avg_precision = sk_metrics.average_precision_score(y_true, y_pred[:, 1])
    _exp.log_metric(prefix + 'avg_precision', avg_precision)

    fig, ax = plt.subplots()
    plt_metrics.plot_precision_recall(y_true, y_pred, ax=ax)
    send_figure(fig, channel_name=prefix + channel_name, experiment=_exp)
    plt.close()


def log_brier_loss(y_true, y_pred_pos, experiment=None, prefix=''):
    """Calculates and logs brier loss to Neptune.

    Args:
        y_true (array-like, shape (n_samples)): Ground truth (correct) target values.
        y_pred_pos (array-like, shape (n_samples)): Score predictions with values from 0 to 1.
        experiment(`neptune.experiments.Experiment`): Neptune experiment. Default is None.
        prefix(str): Prefix that will be added before metric name when logged to Neptune.

    Examples:
        Train the model and make predictions on test::

            from sklearn.datasets import make_classification
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import classification_report

            X, y = make_classification(n_samples=2000)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

            model = RandomForestClassifier()
            model.fit(X_train, y_train)

            y_test_pred = model.predict_proba(X_test)

        Logs Brier score to Neptune::

            import neptune
            from neptunecontrib.monitoring.metrics import log_brier_loss

            neptune.init()
            with neptune.create_experiment():
                log_brier_loss(y_test, y_test_pred[:,1])

        Check out this experiment https://ui.neptune.ai/o/neptune-ai/org/binary-classification-metrics/e/BIN-101/logs.

    """
    assert len(y_pred_pos.shape) == 1, 'y_pred_pos needs to be 1D prediction for positive class'

    _exp = experiment if experiment else neptune

    brier = sk_metrics.brier_score_loss(y_true, y_pred_pos)
    _exp.log_metric(prefix + 'brier_loss', brier)


def log_log_loss(y_true, y_pred, experiment=None, prefix=''):
    """Creates and logs Precision Recall curve and Average precision score to Neptune.

    Args:
        y_true (array-like, shape (n_samples)): Ground truth (correct) target values.
        y_pred (array-like, shape (n_samples, 2)): Predictions for classes 0 and 1 with values from 0 to 1.
        experiment(`neptune.experiments.Experiment`): Neptune experiment. Default is None.
        prefix(str): Prefix that will be added before metric name when logged to Neptune.

    Examples:
        Train the model and make predictions on test::

            from sklearn.datasets import make_classification
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import classification_report

            X, y = make_classification(n_samples=2000)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

            model = RandomForestClassifier()
            model.fit(X_train, y_train)

            y_test_pred = model.predict_proba(X_test)

        Logs log-loss to Neptune::

            import neptune
            from neptunecontrib.monitoring.metrics import log_log_loss

            neptune.init()
            with neptune.create_experiment():
                log_log_loss(y_test, y_test_pred)

        Check out this experiment https://ui.neptune.ai/o/neptune-ai/org/binary-classification-metrics/e/BIN-101/logs.

    """
    assert len(y_pred.shape) == 2, 'y_pred needs to be (n_samples, 2), use expand_prediction helper to format it'

    _exp = experiment if experiment else neptune

    log_loss = sk_metrics.log_loss(y_true, y_pred)
    _exp.log_metric(prefix + 'log_loss', log_loss)


def log_ks_statistic(y_true, y_pred, experiment=None, channel_name='metric_charts', prefix=''):
    """Creates and logs KS statistics curve and KS statistics score to Neptune.

    Kolmogorov-Smirnov statistics chart can be calculated for true positive rates (TPR) and true negative rates (TNR)
    for each threshold and plotted on a chart.
    The maximum distance from TPR to TNR can be treated as performance metric.

    Args:
        y_true (array-like, shape (n_samples)): Ground truth (correct) target values.
        y_pred (array-like, shape (n_samples, 2)): Predictions for classes 0 and 1 with values from 0 to 1.
        experiment(`neptune.experiments.Experiment`): Neptune experiment. Default is None.
        channel_name(str): name of the neptune channel. Default is 'metric_charts'.
        prefix(str): Prefix that will be added before metric name when logged to Neptune.

    Examples:
        Train the model and make predictions on test::

            from sklearn.datasets import make_classification
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import classification_report

            X, y = make_classification(n_samples=2000)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

            model = RandomForestClassifier()
            model.fit(X_train, y_train)

            y_test_pred = model.predict_proba(X_test)

        Create and log KS statistics curve and KS statistics score to Neptune::

            import neptune
            from neptunecontrib.monitoring.metrics import log_ks_statistic

            neptune.init()
            with neptune.create_experiment():
                log_ks_statistic(y_test, y_test_pred)

        Check out this experiment https://ui.neptune.ai/o/neptune-ai/org/binary-classification-metrics/e/BIN-101/logs.

    """
    assert len(y_pred.shape) == 2, 'y_pred needs to be (n_samples, 2), use expand_prediction helper to format it'

    _exp = experiment if experiment else neptune

    res = binary_ks_curve(y_true, y_pred[:, 1])
    ks_stat = res[3]
    _exp.log_metric(prefix + 'ks_statistic', ks_stat)

    fig, ax = plt.subplots()
    plt_metrics.plot_ks_statistic(y_true, y_pred, ax=ax)
    send_figure(fig, channel_name=prefix + channel_name, experiment=_exp)
    plt.close()


def log_cumulative_gain(y_true, y_pred, experiment=None, channel_name='metric_charts', prefix=''):
    """Creates cumulative gain chart and logs it to Neptune.

    Args:
        y_true (array-like, shape (n_samples)): Ground truth (correct) target values.
        y_pred (array-like, shape (n_samples, 2)): Predictions for classes 0 and 1 with values from 0 to 1.
        experiment(`neptune.experiments.Experiment`): Neptune experiment. Default is None.
        channel_name(str): name of the neptune channel. Default is 'metric_charts'.
        prefix(str): Prefix that will be added before metric name when logged to Neptune.

    Examples:
        Train the model and make predictions on test::

            from sklearn.datasets import make_classification
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import classification_report

            X, y = make_classification(n_samples=2000)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

            model = RandomForestClassifier()
            model.fit(X_train, y_train)

            y_test_pred = model.predict_proba(X_test)

        Create and log cumulative gain chart to Neptune::

            import neptune
            from neptunecontrib.monitoring.metrics import log_cumulative_gain

            neptune.init()
            with neptune.create_experiment():
                log_cumulative_gain(y_test, y_test_pred)

        Check out this experiment https://ui.neptune.ai/o/neptune-ai/org/binary-classification-metrics/e/BIN-101/logs.

    """
    assert len(y_pred.shape) == 2, 'y_pred needs to be (n_samples, 2), use expand_prediction helper to format it'

    _exp = experiment if experiment else neptune

    fig, ax = plt.subplots()
    plt_metrics.plot_cumulative_gain(y_true, y_pred, ax=ax)
    send_figure(fig, channel_name=prefix + channel_name, experiment=_exp)
    plt.close()


def log_lift_curve(y_true, y_pred, experiment=None, channel_name='metric_charts', prefix=''):
    """Creates cumulative gain chart and logs it to Neptune.

    Args:
        y_true (array-like, shape (n_samples)): Ground truth (correct) target values.
        y_pred (array-like, shape (n_samples, 2)): Predictions for classes 0 and 1 with values from 0 to 1.
        experiment(`neptune.experiments.Experiment`): Neptune experiment. Default is None.
        channel_name(str): name of the neptune channel. Default is 'metric_charts'.
        prefix(str): Prefix that will be added before metric name when logged to Neptune.

    Examples:
        Train the model and make predictions on test::

            from sklearn.datasets import make_classification
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import classification_report

            X, y = make_classification(n_samples=2000)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

            model = RandomForestClassifier()
            model.fit(X_train, y_train)

            y_test_pred = model.predict_proba(X_test)

        Create and log lift curve chart to Neptune::

            import neptune
            from neptunecontrib.monitoring.metrics import log_lift_curve

            neptune.init()
            with neptune.create_experiment():
                log_lift_curve(y_test, y_test_pred)

        Check out this experiment https://ui.neptune.ai/o/neptune-ai/org/binary-classification-metrics/e/BIN-101/logs.

    """
    assert len(y_pred.shape) == 2, 'y_pred needs to be (n_samples, 2), use expand_prediction helper to format it'

    _exp = experiment if experiment else neptune

    fig, ax = plt.subplots()
    plt_metrics.plot_lift_curve(y_true, y_pred, ax=ax)
    send_figure(fig, channel_name=prefix + channel_name, experiment=_exp)
    plt.close()


def log_prediction_distribution(y_true, y_pred_pos, experiment=None, channel_name='metric_charts', prefix=''):
    """Generates prediction distribution plot from predictions and true labels.

    Args:
        y_true (array-like, shape (n_samples)): Ground truth (correct) target values.
        y_pred_pos (array-like, shape (n_samples)): Score predictions with values from 0 to 1.
        experiment(`neptune.experiments.Experiment`): Neptune experiment. Default is None.
        channel_name(str): name of the neptune channel. Default is 'metric_charts'.
        prefix(str): Prefix that will be added before metric name when logged to Neptune.

    Examples:
        Train the model and make predictions on test::

            from sklearn.datasets import make_classification
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import classification_report

            X, y = make_classification(n_samples=2000)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

            model = RandomForestClassifier()
            model.fit(X_train, y_train)

            y_test_pred = model.predict_proba(X_test)

        Plot prediction distribution::

            from neptunecontrib.monitoring.metrics import log_prediction_distribution

            log_prediction_distribution(y_test, y_test_pred[:, 1])
    """
    assert len(y_pred_pos.shape) == 1, 'y_pred_pos needs to be 1D prediction for positive class'

    _exp = experiment if experiment else neptune

    fig, ax = plt.subplots()
    _plot_prediction_distribution(y_true, y_pred_pos, ax=ax)
    send_figure(fig, channel_name=prefix + channel_name, experiment=_exp)
    plt.close()


def expand_prediction(prediction):
    """Expands 1D binary prediction for positive class.

    Args:
        prediction (array-like, shape (n_samples)):
            Estimated targets as returned by a classifier.

    Returns:
        prediction (array-like, shape (n_samples, 2)):
            Estimated targets for both negative and positive class.
    """
    assert prediction.shape[1] == 2, 'You can only expand 1D prediction for positive classes'

    prediction_reshaped = prediction.reshape(-1, 1)
    return np.clip(np.concatenate((1.0 - prediction_reshaped, prediction_reshaped), axis=1), 0.0, 1.0)


def _plot_confusion_matrix(y_true, y_pred_class, ax=None):
    if not ax:
        _, ax = plt.subplots()
    cmap = plt.get_cmap('Blues')
    cm = sk_metrics.confusion_matrix(y_true, y_pred_class)
    sns.heatmap(cm, cmap=cmap, annot=True, fmt='g', ax=ax)
    ax.set_xlabel('predicted values')
    ax.set_ylabel('actual values')


def _plot_class_metrics_by_threshold(y_true, y_pred_positive):
    scores_by_thres = _class_metrics_by_threshold(y_true, y_pred_positive)
    figs = []
    for name in scores_by_thres.columns:
        if name == 'threshold':
            continue
        else:
            best_thres, best_score = _get_best_thres(scores_by_thres, name)
            fig, ax = plt.subplots()
            ax.plot(scores_by_thres['threshold'], scores_by_thres[name])
            ax.set_title('{} by threshold'.format(name))
            ax.axvline(x=best_thres, color='red')
            ax.text(x=best_thres + 0.01, y=0.98 * best_score,
                    s='thres={:.4f}\nscore={:.4f}'.format(best_thres, best_score),
                    color='red')
            ax.set_xlabel('threshold')
            ax.set_ylabel(name)
            figs.append(fig)
    return figs


def _plot_classification_report(y_true, y_pred_class):
    report = sk_metrics.classification_report(y_true, y_pred_class, output_dict=True)
    report_df = pd.DataFrame(report).transpose().round(4)

    fig, ax = plt.subplots()
    ax.axis('off')
    ax.axis('tight')
    ax.table(cellText=report_df.values,
             colLabels=report_df.columns,
             rowLabels=report_df.index,
             loc='center',
             bbox=[0.2, 0.2, 0.8, 0.8])
    fig.tight_layout()

    return fig


def _plot_prediction_distribution(y_true, y_pred_pos, ax=None):
    if ax is None:
        _, ax = plt.subplots()

    ax.set_title('Prediction Distribution', fontsize='large')

    df = pd.DataFrame({'Prediction': y_pred_pos,
                       'True label': y_true})

    sns.distplot(df[df['True label'] == 0]['Prediction'], label='negative', ax=ax)
    sns.distplot(df[df['True label'] == 1]['Prediction'], label='positive', ax=ax)

    ax.legend(prop={'size': 16}, title='Labels')
    ax.set_xlim([0.0, 1.0])


def _class_metrics(y_true, y_pred_class):
    tn, fp, fn, tp = sk_metrics.confusion_matrix(y_true, y_pred_class).ravel()

    true_positive_rate = tp / (tp + fn)
    true_negative_rate = tn / (tn + fp)
    positive_predictive_value = tp / (tp + fp)
    negative_predictive_value = tn / (tn + fn)
    false_positive_rate = fp / (fp + tn)
    false_negative_rate = fn / (tp + fn)
    false_discovery_rate = fp / (tp + fp)

    scores = {'accuracy': sk_metrics.accuracy_score(y_true, y_pred_class),
              'precision': sk_metrics.precision_score(y_true, y_pred_class),
              'recall': sk_metrics.recall_score(y_true, y_pred_class),
              'f1_score': sk_metrics.fbeta_score(y_true, y_pred_class, beta=1),
              'f2_score': sk_metrics.fbeta_score(y_true, y_pred_class, beta=2),
              'matthews_corrcoef': sk_metrics.matthews_corrcoef(y_true, y_pred_class),
              'cohen_kappa': sk_metrics.cohen_kappa_score(y_true, y_pred_class),
              'true_positive_rate': true_positive_rate,
              'true_negative_rate': true_negative_rate,
              'positive_predictive_value': positive_predictive_value,
              'negative_predictive_value': negative_predictive_value,
              'false_positive_rate': false_positive_rate,
              'false_negative_rate': false_negative_rate,
              'false_discovery_rate': false_discovery_rate}

    return scores


def _class_metrics_by_threshold(y_true, y_pred_pos, thres_nr=100):
    thresholds = [i / thres_nr for i in range(1, thres_nr, 1)]

    scores_per_thres = []
    for thres in thresholds:
        y_pred_class = y_pred_pos > thres
        scores = _class_metrics(y_true, y_pred_class)
        scores['threshold'] = thres
        scores_per_thres.append(pd.Series(scores))

    return pd.DataFrame(scores_per_thres)


def _get_best_thres(scores_by_thres, name):
    if name in ['false_positive_rate', 'false_negative_rate', 'false_discovery_rate']:
        best_res = scores_by_thres[scores_by_thres[name] == scores_by_thres[name].min()][['threshold', name]]
    else:
        best_res = scores_by_thres[scores_by_thres[name] == scores_by_thres[name].max()][['threshold', name]]
    position = len(best_res) // 2
    result = best_res.iloc[position].to_dict()
    return result['threshold'], result[name]
