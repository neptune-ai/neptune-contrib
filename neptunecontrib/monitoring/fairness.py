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
from aif360.datasets import BinaryLabelDataset
from aif360.metrics.classification_metric import ClassificationMetric
import neptune
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from neptunecontrib.monitoring.utils import send_figure


def log_fairness_classification_metrics(y_true, y_pred_class, y_pred_score, sensitive_attributes,
                                        favorable_label, unfavorable_label,
                                        privileged_groups, unprivileged_groups,
                                        experiment=None, prefix=''):
    """Creates fairness metric charts, calculates fairness classification metrics and logs them to Neptune.

    Class-based metrics that are logged: 'true_positive_rate_difference','false_positive_rate_difference',
    'false_omission_rate_difference', 'false_discovery_rate_difference', 'error_rate_difference',
    'false_positive_rate_ratio', 'false_negative_rate_ratio', 'false_omission_rate_ratio',
    'false_discovery_rate_ratio', 'error_rate_ratio', 'average_odds_difference', 'disparate_impact',
    'statistical_parity_difference', 'equal_opportunity_difference', 'theil_index',
    'between_group_theil_index', 'between_all_groups_theil_index', 'coefficient_of_variation',
    'between_group_coefficient_of_variation', 'between_all_groups_coefficient_of_variation',
    'generalized_entropy_index', 'between_group_generalized_entropy_index',
    'between_all_groups_generalized_entropy_index'

    Charts are logged to the 'metric_by_group' channel: 'confusion matrix', 'TPR', 'TNR', 'FPR', 'FNR', 'PPV', 'NPV',
    'FDR', 'FOR', 'ACC', 'error_rate', 'selection_rate', 'power', 'precision', 'recall',
    'sensitivity', 'specificity'.

    Args:
        y_true (array-like, shape (n_samples)): Ground truth (correct) target values.
        y_pred_class (array-like, shape (n_samples)): Class predictions with values 0 or 1.
        y_pred_score (array-like, shape (n_samples)): Class predictions with values from 0 to 1. Default None.
        sensitive_attributes (pandas.DataFrame, shape (n_samples, k)): datafame containing only sensitive columns.
        favorable_label (str or int): label that is favorable, brings positive value to a person being classified.
        unfavorable_label (str or int): label that is unfavorable, brings positive value to a person being classified.
        privileged_groups (dict): dictionary with column names and list of values for those columns that
           belong to the privileged groups.
        unprivileged_groups (dict): dictionary with column names and list of values for those columns that
           belong to the unprivileged groups.
        experiment(`neptune.experiments.Experiment`): Neptune experiment. Default is None.
        prefix(str): Prefix that will be added before metric name when logged to Neptune.

    Examples:
        Train the model and make predictions on test.
        Log metrics and performance curves to Neptune::

            import neptune
            from neptunecontrib.monitoring.fairness import log_fairness_classification_metrics

            neptune.init()
            with neptune.create_experiment():
                log_fairness_classification_metrics(y_true, y_pred_class, y_pred_score, test[['race']],
                                                    favorable_label='granted_parole',
                                                    unfavorable_label='not_granted_parole',
                                                    privileged_groups={'race':['Caucasian']},
                                                    privileged_groups={'race':['African-American','Hispanic]},
                                                    )

        Check out this experiment https://ui.neptune.ai/jakub-czakon/model-fairness/e/MOD-92/logs.

    """
    _exp = experiment if experiment else neptune

    bias_info = {'favorable_label': favorable_label,
                 'unfavorable_label': unfavorable_label,
                 'protected_columns': sensitive_attributes.columns.tolist()}

    privileged_info = _fmt_priveleged_info(privileged_groups, unprivileged_groups)

    ground_truth_test = _make_dataset(sensitive_attributes, y_true, **bias_info, **privileged_info)
    prediction_test = _make_dataset(sensitive_attributes, y_pred_class, y_pred_score, **bias_info, **privileged_info)

    clf_metric = ClassificationMetric(ground_truth_test, prediction_test, **privileged_info)

    _log_fairness_metrics(clf_metric, _exp, prefix)

    fig = _plot_confusion_matrix_by_group(clf_metric, figsize=(12, 4))
    plt.tight_layout()
    plt.close()
    send_figure(fig, channel_name=prefix + 'metrics_by_group')

    group_metrics = ['TPR', 'TNR', 'FPR', 'FNR', 'PPV', 'NPV', 'FDR', 'FOR',
                     'ACC', 'error_rate', 'selection_rate', 'power',
                     'precision', 'recall', 'sensitivity', 'specificity']

    for metric_name in group_metrics:
        fig, ax = plt.subplots(figsize=(12, 8))
        _plot_performance_by_group(clf_metric, metric_name, ax)
        send_figure(fig, experiment=_exp, channel_name=prefix + 'metrics_by_group')
        plt.close()


def _make_dataset(features, labels, scores=None, protected_columns=None,
                  privileged_groups=None, unprivileged_groups=None,
                  favorable_label=None, unfavorable_label=None):
    df = features.copy()
    df['outcome'] = labels

    if scores is not None:
        scores_names = 'scores'
        df[scores_names] = scores
    else:
        scores_names = []

    dataset = BinaryLabelDataset(df=df, label_names=['outcome'], scores_names=scores_names,
                                 protected_attribute_names=protected_columns,
                                 favorable_label=favorable_label, unfavorable_label=unfavorable_label,
                                 unprivileged_protected_attributes=unprivileged_groups)
    return dataset


def _fmt_priveleged_info(privileged_groups, unprivileged_groups):
    privileged_info = {}
    for name, group in zip(['privileged_groups', 'unprivileged_groups'],
                           [privileged_groups, unprivileged_groups]):
        privileged_info[name] = []
        for k, values in group.items():
            for v in values:
                privileged_info[name].append({k: v})

    return privileged_info


def _log_fairness_metrics(aif_metric, experiment, prefix):
    func_dict = {
        'true_positive_rate_difference': aif_metric.true_positive_rate_difference,
        'false_positive_rate_difference': aif_metric.false_positive_rate_difference,
        'false_omission_rate_difference': aif_metric.false_omission_rate_difference,
        'false_discovery_rate_difference': aif_metric.false_discovery_rate_difference,
        'error_rate_difference': aif_metric.error_rate_difference,

        'false_positive_rate_ratio': aif_metric.false_positive_rate_ratio,
        'false_negative_rate_ratio': aif_metric.false_negative_rate_ratio,
        'false_omission_rate_ratio': aif_metric.false_omission_rate_ratio,
        'false_discovery_rate_ratio': aif_metric.false_discovery_rate_ratio,
        'error_rate_ratio': aif_metric.error_rate_ratio,

        'average_odds_difference': aif_metric.average_odds_difference,

        'disparate_impact': aif_metric.disparate_impact,
        'statistical_parity_difference': aif_metric.statistical_parity_difference,
        'equal_opportunity_difference': aif_metric.equal_opportunity_difference,
        'theil_index': aif_metric.theil_index,
        'between_group_theil_index': aif_metric.between_group_theil_index,
        'between_all_groups_theil_index': aif_metric.between_all_groups_theil_index,
        'coefficient_of_variation': aif_metric.coefficient_of_variation,
        'between_group_coefficient_of_variation': aif_metric.between_group_coefficient_of_variation,
        'between_all_groups_coefficient_of_variation': aif_metric.between_all_groups_coefficient_of_variation,

        'generalized_entropy_index': aif_metric.generalized_entropy_index,
        'between_group_generalized_entropy_index': aif_metric.between_group_generalized_entropy_index,
        'between_all_groups_generalized_entropy_index': aif_metric.between_all_groups_generalized_entropy_index}

    for name, func in func_dict.items():
        score = func()
        experiment.log_metric(prefix + name, score)


def _plot_confusion_matrix_by_group(aif_metric, figsize=None):
    if not figsize:
        figsize = (18, 4)

    cmap = plt.get_cmap('Blues')
    fig, axs = plt.subplots(1, 3, figsize=figsize)

    axs[0].set_title('all')
    cm = _format_aif360_to_sklearn(aif_metric.binary_confusion_matrix(privileged=None))
    sns.heatmap(cm, cmap=cmap, annot=True, fmt='g', ax=axs[0])
    axs[0].set_xlabel('predicted values')
    axs[0].set_ylabel('actual values')

    axs[1].set_title('privileged')
    cm = _format_aif360_to_sklearn(aif_metric.binary_confusion_matrix(privileged=True))
    sns.heatmap(cm, cmap=cmap, annot=True, fmt='g', ax=axs[1])
    axs[1].set_xlabel('predicted values')
    axs[1].set_ylabel('actual values')

    axs[2].set_title('unprivileged')
    cm = _format_aif360_to_sklearn(aif_metric.binary_confusion_matrix(privileged=False))
    sns.heatmap(cm, cmap=cmap, annot=True, fmt='g', ax=axs[2])
    axs[2].set_xlabel('predicted values')
    axs[2].set_ylabel('actual values')
    return fig


def _plot_performance_by_group(aif_metric, metric_name, ax=None):
    performance_metrics = ['TPR', 'TNR', 'FPR', 'FNR', 'PPV', 'NPV', 'FDR', 'FOR', 'ACC']

    func_dict = {'selection_rate': lambda x: aif_metric.selection_rate(privileged=x),
                 'precision': lambda x: aif_metric.precision(privileged=x),
                 'recall': lambda x: aif_metric.recall(privileged=x),
                 'sensitivity': lambda x: aif_metric.sensitivity(privileged=x),
                 'specificity': lambda x: aif_metric.specificity(privileged=x),
                 'power': lambda x: aif_metric.power(privileged=x),
                 'error_rate': lambda x: aif_metric.error_rate(privileged=x)}

    if not ax:
        _, ax = plt.subplots()

    if metric_name in performance_metrics:
        metric_func = lambda x: aif_metric.performance_measures(privileged=x)[metric_name]
    elif metric_name in func_dict.keys():
        metric_func = func_dict[metric_name]
    else:
        raise NotImplementedError

    df = pd.DataFrame()
    df['Group'] = ['all', 'priveleged', 'unpriveleged']
    df[metric_name] = [metric_func(group) for group in [None, True, False]]

    sns.barplot(x='Group', y=metric_name, data=df, ax=ax)
    ax.set_title('{} by group'.format(metric_name))
    ax.set_xlabel(None)

    _add_annotations(ax)


def _add_annotations(ax):
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.3f'),
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center',
                    xytext=(0, -10), textcoords='offset points')


def _format_aif360_to_sklearn(aif360_mat):
    return np.array([[aif360_mat['TN'], aif360_mat['FP']],
                     [aif360_mat['FN'], aif360_mat['TP']]])
