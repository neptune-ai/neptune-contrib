#
# Copyright (c) 2020, Neptune Labs Sp. z o.o.
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
import pandas as pd
from scikitplot.estimators import plot_learning_curve
from scikitplot.metrics import plot_precision_recall
from sklearn.base import is_regressor, is_classifier
from sklearn.metrics import explained_variance_score, max_error, mean_absolute_error, r2_score, \
    precision_recall_fscore_support
from sklearn.utils import estimator_html_repr
from yellowbrick.classifier import ClassificationReport, ConfusionMatrix, ROCAUC, ClassPredictionError
from yellowbrick.cluster import SilhouetteVisualizer, KElbowVisualizer
from yellowbrick.model_selection import FeatureImportances
from yellowbrick.regressor import ResidualsPlot, PredictionError, CooksDistance

from neptunecontrib.api.html import log_html
from neptunecontrib.api.table import log_csv
from neptunecontrib.api.utils import log_pickle


def log_regressor_summary(regressor,
                          X_train,
                          X_test,
                          y_train,
                          y_test,
                          experiment=None,
                          log_params=True,
                          log_model=True,
                          log_test_preds=True,
                          log_test_scores=True,
                          log_visualizations=True):
    """Log sklearn regressor summary.

    This method automatically logs all regressor parameters, pickled estimator (model),
    test predictions as table, model performance visualizations,
    sklearn's pipeline as an interactive graph and test metrics.

    Regressor must be fitted before calling this function.

    Make sure you created an experiment before you use this method: ``neptune.create_experiment()``.

    Tip:
        Check `Neptune documentation <https://docs.neptune.ai/integrations/scikit_learn.html>`_ for the full example.

    Args:
        regressor (:obj:`regressor`):
            | Fitted sklearn regressor object
        X_train (:obj:`ndarray`):
            | Training data matrix
        X_test (:obj:`ndarray`):
            | Testing data matrix
        y_train (:obj:`ndarray`):
            | The regression target for training
        y_test (:obj:`ndarray`):
            | The regression target for testing
        experiment (:obj:`neptune.experiments.Experiment`, optional, default is ``None``):
            | Neptune ``Experiment`` object to control to which experiment you log the data.
            | If ``None``, log to currently active, and most recent experiment.
        log_params (:obj:`bool`, optional, default is ``True``):
            | Log regressor parameters as properties.
        log_model (:obj:`bool`, optional, default is ``True``):
            | Log pickled regressor.
        log_test_preds (:obj:`bool`, optional, default is ``True``):
            | Log test predictions as csv file.
        log_test_scores (:obj:`bool`, optional, default is ``True``):
            | Log suite of test scores.
        log_visualizations (:obj:`bool`, optional, default is ``True``):
            | Log suite of the regressor visualizations'.

    Returns:
        ``None``

    Examples:
        Log random forest regressor summary

        .. code:: python3

            rfr = RandomForestRegressor()

            X, y = load_boston(return_X_y=True)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

            rfr.fit(X_train, y_train)

            neptune.init('shared/sklearn-integration')
            neptune.create_experiment(name='regression-example',
                                      tags=['RandomForestRegressor', 'regression'])

            log_regressor_summary(rfr, X_train, X_test, y_train, y_test)
    """
    exp = _check_experiment(experiment)
    _check_estimator(regressor, 'regressor')

    y_pred = _compute_test_preds(regressor, X_test)

    _log_estimator_params(log_params, regressor, exp)
    _log_pickled_model(log_model, regressor, exp)
    _log_test_predictions(log_test_preds, y_pred, y_test, exp)

    if log_test_scores:
        # single output
        if len(y_pred.shape) == 1:
            evs = explained_variance_score(y_test, y_pred)
            me = max_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            exp.log_metric('evs_sklearn', evs)
            exp.log_metric('me_sklearn', me)
            exp.log_metric('mae_sklearn', mae)
            exp.log_metric('r2_sklearn', r2)

        # multi output
        if len(y_pred.shape) == 2:
            r2 = regressor.score(X_test, y_test)
            exp.log_metric('r2_sklearn', r2)

    if log_visualizations:
        try:
            fig, ax = plt.subplots()
            plot_learning_curve(regressor, X_train, y_train, ax=ax)
            exp.log_image('charts_sklearn', fig, image_name='Learning Curve')
        except Exception:
            print('Did not log learning curve chart.')

        try:
            log_html('estimator_visualization', estimator_html_repr(regressor), exp)
        except Exception:
            print('Did not log estimator visualization as html.')

        try:
            fig, ax = plt.subplots()
            visualizer = FeatureImportances(regressor, is_fitted=True, ax=ax)
            visualizer.fit(X_train, y_train)
            visualizer.finalize()
            exp.log_image('charts_sklearn', fig, image_name='Feature Importance')
        except Exception:
            print('Did not log feature importance chart.')

        try:
            fig, ax = plt.subplots()
            visualizer = ResidualsPlot(regressor, is_fitted=True, ax=ax)
            visualizer.fit(X_train, y_train)
            visualizer.score(X_test, y_test)
            visualizer.finalize()
            exp.log_image('charts_sklearn', fig, image_name='Residuals Plot')
        except Exception:
            print('Did not log residuals plot chart.')

        try:
            fig, ax = plt.subplots()
            visualizer = PredictionError(regressor, is_fitted=True, ax=ax)
            visualizer.fit(X_train, y_train)
            visualizer.score(X_test, y_test)
            visualizer.finalize()
            exp.log_image('charts_sklearn', fig, image_name='Prediction Error')
        except Exception:
            print('Did not log prediction error chart.')

        try:
            fig, ax = plt.subplots()
            visualizer = CooksDistance(ax=ax)
            visualizer.fit(X_train, y_train)
            visualizer.finalize()
            exp.log_image('charts_sklearn', fig, image_name='Cooks Distance')
        except Exception:
            print('Did not log cooks distance chart.')

        plt.close('all')


def log_classifier_summary(classifier,
                           X_train,
                           X_test,
                           y_train,
                           y_test,
                           experiment=None,
                           log_params=True,
                           log_model=True,
                           log_test_preds=True,
                           log_test_scores=True,
                           log_visualizations=True):
    """Log sklearn classifier summary.

    This method automatically logs all classifier parameters, pickled estimator (model),
    test predictions, predictions probabilities as table, model performance visualizations,
    sklearn's pipeline as an interactive graph and test metrics.

    Classifier must be fitted before calling this function.

    Make sure you created an experiment before you use this method: ``neptune.create_experiment()``.

    Tip:
        Check `Neptune documentation <https://docs.neptune.ai/integrations/scikit_learn.html>`_ for the full example.

    Args:
        classifier (:obj:`classifier`):
            | Fitted sklearn classifier object
        X_train (:obj:`ndarray`):
            | Training data matrix
        X_test (:obj:`ndarray`):
            | Testing data matrix
        y_train (:obj:`ndarray`):
            | The classification target for training
        y_test (:obj:`ndarray`):
            | The classification target for testing
        experiment (:obj:`neptune.experiments.Experiment`, optional, default is ``None``):
            | Neptune ``Experiment`` object to control to which experiment you log the data.
            | If ``None``, log to currently active, and most recent experiment.
        log_params (:obj:`bool`, optional, default is ``True``):
            | Log classifier parameters as properties.
        log_model (:obj:`bool`, optional, default is ``True``):
            | Log pickled classifier.
        log_test_preds (:obj:`bool`, optional, default is ``True``):
            | Log test predictions as csv file.
        log_test_scores (:obj:`bool`, optional, default is ``True``):
            | Log suite of test scores.
        log_visualizations (:obj:`bool`, optional, default is ``True``):
            | Log suite of the classifier visualizations'.

    Returns:
        ``None``

    Examples:
        Log random forest classifier summary

        .. code:: python3

            rfr = RandomForestClassifier()

            X, y = load_boston(return_X_y=True)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

            rfr.fit(X_train, y_train)

            neptune.init('shared/sklearn-integration')
            neptune.create_experiment(name='classification-example',
                                      tags=['RandomForestClassifier', 'classification'])

            log_classifier_summary(rfr, X_train, X_test, y_train, y_test)
    """
    exp = _check_experiment(experiment)
    _check_estimator(classifier, 'classifier')

    (y_pred, y_pred_proba) = _compute_test_preds(classifier, X_test)

    _log_estimator_params(log_params, classifier, exp)
    _log_pickled_model(log_model, classifier, exp)
    _log_test_predictions(log_test_preds, y_pred, y_test, exp)
    _log_test_predictions_probabilities(log_test_preds, classifier, y_pred_proba, exp)

    if log_test_scores:
        for name, values in zip(['precision', 'recall', 'fbeta_score', 'support'],
                                precision_recall_fscore_support(y_test, y_pred)):
            for i, value in enumerate(values):
                exp.log_metric('{}_class_{}_sklearn'.format(name, i), value)

    if log_visualizations:
        try:
            fig, ax = plt.subplots()
            visualizer = ClassificationReport(classifier, support=True, is_fitted=True, ax=ax)
            visualizer.fit(X_train, y_train)
            visualizer.score(X_test, y_test)
            visualizer.finalize()
            exp.log_image('charts_sklearn', fig, image_name='Classification Report')
        except Exception:
            print('Did not log Classification Report chart.')

        try:
            fig, ax = plt.subplots()
            visualizer = ConfusionMatrix(classifier, is_fitted=True, ax=ax)
            visualizer.fit(X_train, y_train)
            visualizer.score(X_test, y_test)
            visualizer.finalize()
            exp.log_image('charts_sklearn', fig, image_name='Confusion Matrix')
        except Exception:
            print('Did not log Confusion Matrix chart.')

        try:
            fig, ax = plt.subplots()
            visualizer = ROCAUC(classifier, is_fitted=True, ax=ax)
            visualizer.fit(X_train, y_train)
            visualizer.score(X_test, y_test)
            visualizer.finalize()
            exp.log_image('charts_sklearn', fig, image_name='ROC-AUC')
        except Exception:
            print('Did not log ROC-AUC chart.')

        try:
            fig, ax = plt.subplots()
            plot_precision_recall(y_test, y_pred_proba, ax=ax)
            exp.log_image('charts_sklearn', fig, image_name='Precision Recall Curve')
        except Exception:
            print('Did not log Precision-Recall chart.')

        try:
            fig, ax = plt.subplots()
            visualizer = ClassPredictionError(classifier, is_fitted=True, ax=ax)
            visualizer.fit(X_train, y_train)
            visualizer.score(X_test, y_test)
            visualizer.finalize()
            exp.log_image('charts_sklearn', fig, image_name='Class Prediction Error')
        except Exception:
            print('Did not log Class Prediction Error chart.')

        try:
            log_html('estimator_visualization', estimator_html_repr(classifier), exp)
        except Exception:
            print('Did not log estimator visualization as html.')

        plt.close('all')


def log_kmeans_clustering_summary(model,
                                  data,
                                  k=10,
                                  experiment=None,
                                  log_params=True,
                                  log_cluster_labels=True,
                                  log_visualizations=True):
    """Log sklearn clustering summary.

    This method automatically logs all clustering parameters, cluster labels on data,
    sklearn's pipeline as an interactive graph and clustering visualizations.

    Make sure you created an experiment before you use this method: ``neptune.create_experiment()``.

    Tip:
        Check `Neptune documentation <https://docs.neptune.ai/integrations/scikit_learn.html>`_ for the full example.

    Args:
        model (:obj:`KMeans`):
            | KMeans object
        data (:obj:`ndarray`):
            | Training instances to cluster
        k (`int`):
            | Number of clusters
        experiment (:obj:`neptune.experiments.Experiment`, optional, default is ``None``):
            | Neptune ``Experiment`` object to control to which experiment you log the data.
            | If ``None``, log to currently active, and most recent experiment.
        log_params (:obj:`bool`, optional, default is ``True``):
            | Log kmeans parameters as properties.
        log_cluster_labels (:obj:`bool`, optional, default is ``True``):
            | Log the index of the cluster each sample belongs to.
        log_visualizations (:obj:`bool`, optional, default is ``True``):
            | Log suite of the clustering visualizations'.

    Returns:
        ``None``

    Examples:
        Log kmeans clustering summary

        .. code:: python3

            km = KMeans(n_init=11, max_iter=270)

            X, y = make_blobs(n_samples=579, n_features=17, centers=7, random_state=28743)

            neptune.init('shared/sklearn-integration')
            neptune.create_experiment(name='clustering-example',
                                      tags=['KMeans', 'clustering'])

            log_kmeans_clustering_summary(km, data=X, k=11)
    """
    exp = _check_experiment(experiment)

    model.set_params(n_clusters=k)
    labels = model.fit_predict(data)

    _log_estimator_params(log_params, model, exp)

    if log_cluster_labels:
        df = pd.DataFrame(data={'cluster_labels': labels})
        log_csv('cluster_labels', df, experiment)

    if log_visualizations:
        try:
            log_html('estimator_visualization', estimator_html_repr(model), exp)
        except Exception:
            print('Did not log estimator visualization as html.')

        try:
            fig, ax = plt.subplots()
            visualizer = KElbowVisualizer(model, ax=ax)
            visualizer.fit(data)
            visualizer.finalize()
            exp.log_image('charts_sklearn', fig, image_name='Class Prediction Error')
        except Exception:
            print('Did not log Class Prediction Error chart.')

        for j in range(2, k+1):
            model.set_params(n_clusters=j)
            model.fit(data)

            try:
                fig, ax = plt.subplots()
                visualizer = SilhouetteVisualizer(model, is_fitted=True, ax=ax)
                visualizer.fit(data)
                visualizer.finalize()
                exp.log_image('charts_sklearn', fig, image_name='Silhouette Coefficients for k={}'.format(j))
            except Exception:
                print('Did not log Silhouette Coefficients chart.')

        plt.close('all')


def _check_experiment(experiment):
    if experiment is not None:
        if not isinstance(experiment, neptune.experiments.Experiment):
            ValueError('Passed experiment is not Neptune experiment. Create one by using "create_experiment()"')
    else:
        try:
            experiment = neptune.get_experiment()
        except neptune.exceptions.NeptuneNoExperimentContextException:
            raise neptune.exceptions.NeptuneNoExperimentContextException()

    return experiment


def _check_estimator(estimator, estimator_type):
    if estimator_type == 'regressor' and not is_regressor(estimator):
        raise ValueError('"regressor" is not sklearn regressor. This method works only with sklearn regressors.')
    if estimator_type == 'classifier' and not is_classifier(estimator):
        raise ValueError('"classifier" is not sklearn classifier. This method works only with sklearn classifiers.')


def _compute_test_preds(estimator, data):
    if is_regressor(estimator):
        return estimator.predict(data)
    if is_classifier(estimator):
        y_pred = estimator.predict(data)
        y_pred_proba = None
        try:
            y_pred_proba = estimator.predict_proba(data)
        except Exception:
            print('This classifier does not provide predictions probabilities.')
        return y_pred, y_pred_proba


def _log_estimator_params(flag, estimator, experiment):
    if flag:
        for param, value in estimator.get_params().items():
            experiment.set_property(param, value)


def _log_pickled_model(flag, estimator, experiment):
    if flag:
        log_pickle('model/estimator.skl', estimator, experiment)


def _log_test_predictions(flag, y_pred, y_test, experiment):
    if flag:
        # single output
        if len(y_pred.shape) == 1:
            df = pd.DataFrame(data={'y_true': y_test, 'y_pred': y_pred})
            log_csv('test_predictions', df, experiment)

        # multi output
        if len(y_pred.shape) == 2:
            df = pd.DataFrame()
            for j in range(y_pred.shape[1]):
                df['y_test_output_{}'.format(j)] = y_test[:, j]
                df['y_pred_output_{}'.format(j)] = y_pred[:, j]
            log_csv('test_predictions', df, experiment)


def _log_test_predictions_probabilities(flag, classifier, y_pred_proba, experiment):
    if flag and y_pred_proba is not None:
        df = pd.DataFrame(data=y_pred_proba, columns=classifier.classes_)
        log_csv('test_preds_proba', df, experiment)
