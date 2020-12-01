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
from sklearn.cluster import KMeans


def log_regressor_summary(regressor,
                          X_train,
                          X_test,
                          y_train,
                          y_test,
                          experiment=None,
                          log_visualizations=True):
    """Log sklearn regressor summary.

    This method automatically logs all regressor parameters, pickled estimator (model),
    test predictions as table, model performance visualizations,
    sklearn's pipeline as an interactive graph and test metrics.

    Regressor should be fitted before calling this function.

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
    assert is_regressor(regressor), 'regressor should be sklearn regressor.'

    exp = _validate_experiment(experiment)

    log_estimator_params(regressor, exp)
    log_pickled_model(regressor, exp)

    y_pred = regressor.predict(X_test)
    log_test_predictions(regressor, X_test, y_test, y_pred=y_pred, experiment=exp)
    log_test_scores(regressor, X_test, y_test, y_pred=y_pred, experiment=exp)

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
                           log_visualizations=True):
    """Log sklearn classifier summary.

    This method automatically logs all classifier parameters, pickled estimator (model),
    test predictions, predictions probabilities as table, model performance visualizations,
    sklearn's pipeline as an interactive graph and test metrics.

    Classifier should be fitted before calling this function.

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
    assert is_classifier(classifier), 'classifier should be sklearn classifier.'

    exp = _validate_experiment(experiment)

    log_estimator_params(classifier, exp)
    log_pickled_model(classifier, exp)
    log_test_predictions_probabilities(classifier, X_test, experiment=exp)

    y_pred = classifier.predict(X_test)
    log_test_predictions(classifier, X_test, y_test, y_pred=y_pred, experiment=exp)
    log_test_scores(classifier, X_test, y_test, y_pred=y_pred, experiment=exp)

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
                                  X,
                                  k=10,
                                  experiment=None,
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
        X (:obj:`ndarray`):
            | Training instances to cluster
        k (`int`):
            | Number of clusters
        experiment (:obj:`neptune.experiments.Experiment`, optional, default is ``None``):
            | Neptune ``Experiment`` object to control to which experiment you log the data.
            | If ``None``, log to currently active, and most recent experiment.
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

            log_kmeans_clustering_summary(km, X=X, k=11)
    """
    assert isinstance(model, KMeans), 'model should be sklearn KMeans instance'

    exp = _validate_experiment(experiment)

    model.set_params(n_clusters=k)
    labels = model.fit_predict(X)

    log_estimator_params(model, exp)

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
            visualizer.fit(X)
            visualizer.finalize()
            exp.log_image('charts_sklearn', fig, image_name='Class Prediction Error')
        except Exception:
            print('Did not log Class Prediction Error chart.')

        for j in range(2, k+1):
            model.set_params(n_clusters=j)
            model.fit(X)

            try:
                fig, ax = plt.subplots()
                visualizer = SilhouetteVisualizer(model, is_fitted=True, ax=ax)
                visualizer.fit(X)
                visualizer.finalize()
                exp.log_image('charts_sklearn', fig, image_name='Silhouette Coefficients for k={}'.format(j))
            except Exception:
                print('Did not log Silhouette Coefficients chart.')

        plt.close('all')


def log_estimator_params(estimator, experiment=None):
    """Log estimator parameters

    Log all estimator parameters as experiment properties.

    Make sure you created an experiment before you use this method: ``neptune.create_experiment()``.

    Tip:
        Check `Neptune documentation <https://docs.neptune.ai/integrations/scikit_learn.html>`_ for the full example.

    Args:
        estimator (:obj:`estimator`):
            | Scikit-learn estimator from which to log parameters.
        experiment (:obj:`neptune.experiments.Experiment`, optional, default is ``None``):
            | Neptune ``Experiment`` object to control to which experiment you log the data.
            | If ``None``, log to currently active, and most recent experiment.

    Returns:
        ``None``

    Examples:
        .. code:: python3

            rfr = RandomForestRegressor()
            log_estimator_params(rfr)
    """
    assert is_regressor(estimator) or is_classifier(estimator) or isinstance(estimator, KMeans),\
        'Estimator should be sklearn regressor, classifier or kmeans clusterer.'

    exp = _validate_experiment(experiment)

    for param, value in estimator.get_params().items():
        exp.set_property(param, value)


def log_pickled_model(estimator, experiment=None):
    """Log pickled estimator.

    Log estimator as pickled file to Neptune artifacts.

    Path to file in the Neptune artifacts is 'model/estimator.skl'.

    Make sure you created an experiment before you use this method: ``neptune.create_experiment()``.

    Tip:
        Check `Neptune documentation <https://docs.neptune.ai/integrations/scikit_learn.html>`_ for the full example.

    Args:
        estimator (:obj:`estimator`):
            | Scikit-learn estimator to log.
        experiment (:obj:`neptune.experiments.Experiment`, optional, default is ``None``):
            | Neptune ``Experiment`` object to control to which experiment you log the data.
            | If ``None``, log to currently active, and most recent experiment.

    Returns:
        ``None``

    Examples:
        .. code:: python3

            rfr = RandomForestRegressor()
            log_pickled_model(rfr, my_experiment)
    """
    assert is_regressor(estimator) or is_classifier(estimator),\
        'Estimator should be sklearn regressor or classifier.'

    exp = _validate_experiment(experiment)

    log_pickle('model/estimator.skl', estimator, exp)


def log_test_predictions(estimator, X_test, y_test, y_pred=None, experiment=None):
    """Log test predictions.

    Calculate and log test predictions and have them as csv file in the Neptune artifacts.

    If you pass ``y_pred``, then predictions are logged without computing from ``X_test`` data.

    Path to predictions in the Neptune artifacts is 'csv/test_predictions.csv'.

    Make sure you created an experiment before you use this method: ``neptune.create_experiment()``.

    Tip:
        Check `Neptune documentation <https://docs.neptune.ai/integrations/scikit_learn.html>`_ for the full example.

    Args:
        estimator (:obj:`estimator`):
            | Scikit-learn estimator to compute predictions.
        X_test (:obj:`ndarray`):
            | Testing data matrix.
        y_test (:obj:`ndarray`):
            | Target for testing.
        y_pred (:obj:`ndarray`, optional, default is ``None``):
            | Estimator predictions on test data.
        experiment (:obj:`neptune.experiments.Experiment`, optional, default is ``None``):
            | Neptune ``Experiment`` object to control to which experiment you log the data.
            | If ``None``, log to currently active, and most recent experiment.

    Returns:
        ``None``

    Examples:
        .. code:: python3

            rfr = RandomForestRegressor()
            log_test_predictions(rfr, X_test, y_test)
    """
    assert is_regressor(estimator) or is_classifier(estimator),\
        'Estimator should be sklearn regressor or classifier.'

    exp = _validate_experiment(experiment)

    if y_pred is None:
        y_pred = estimator.predict(X_test)

    # single output
    if len(y_pred.shape) == 1:
        df = pd.DataFrame(data={'y_true': y_test, 'y_pred': y_pred})
        log_csv('test_predictions', df, exp)

    # multi output
    if len(y_pred.shape) == 2:
        df = pd.DataFrame()
        for j in range(y_pred.shape[1]):
            df['y_test_output_{}'.format(j)] = y_test[:, j]
            df['y_pred_output_{}'.format(j)] = y_pred[:, j]
        log_csv('test_predictions', df, exp)


def log_test_predictions_probabilities(classifier, X_test, y_pred_proba=None, experiment=None):
    """Log test predictions probabilities.

    Calculate and log test predictions probabilities and have them as csv file in the Neptune artifacts.

    If you pass ``y_pred_proba``, then predictions probabilities are logged without computing from ``X_test`` data.

    Path to predictions probabilities in the Neptune artifacts is 'csv/test_preds_proba.csv'.

    Make sure you created an experiment before you use this method: ``neptune.create_experiment()``.

    Tip:
        Check `Neptune documentation <https://docs.neptune.ai/integrations/scikit_learn.html>`_ for the full example.

    Args:
        classifier (:obj:`classifier`):
            | Scikit-learn classifier to compute predictions probabilities.
        X_test (:obj:`ndarray`):
            | Testing data matrix.
        y_pred_proba (:obj:`ndarray`, optional, default is ``None``):
            | Classifier predictions probabilities on test data.
        experiment (:obj:`neptune.experiments.Experiment`, optional, default is ``None``):
            | Neptune ``Experiment`` object to control to which experiment you log the data.
            | If ``None``, log to currently active, and most recent experiment.

    Returns:
        ``None``

    Examples:
        .. code:: python3

            rfc = RandomForestClassifier()
            log_test_predictions(rfc, X_test, y_test)
    """
    assert is_classifier(classifier), 'Classifier should be sklearn classifier.'

    exp = _validate_experiment(experiment)

    if y_pred_proba is None:
        try:
            y_pred_proba = classifier.predict_proba(X_test)
        except Exception:
            print('This classifier does not provide predictions probabilities.')
            return

    df = pd.DataFrame(data=y_pred_proba, columns=classifier.classes_)
    log_csv('test_preds_proba', df, exp)


def log_test_scores(estimator, X_test, y_test, y_pred=None, experiment=None):
    """Log test scores.

    Calculate and log scores on test data and have them as metrics in Neptune.

    If you pass ``y_pred``, then predictions are not computed from ``X_test`` data.

    Make sure you created an experiment before you use this method: ``neptune.create_experiment()``.

    Tip:
        Check `Neptune documentation <https://docs.neptune.ai/integrations/scikit_learn.html>`_ for the full example.

    Args:
        estimator (:obj:`estimator`):
            | Scikit-learn estimator to compute scores.
        X_test (:obj:`ndarray`):
            | Testing data matrix.
        y_test (:obj:`ndarray`):
            | Target for testing.
        y_pred (:obj:`ndarray`, optional, default is ``None``):
            | Estimator predictions on test data.
        experiment (:obj:`neptune.experiments.Experiment`, optional, default is ``None``):
            | Neptune ``Experiment`` object to control to which experiment you log the data.
            | If ``None``, log to currently active, and most recent experiment.

    Returns:
        ``None``

    Examples:
        .. code:: python3

            rfc = RandomForestClassifier()
            log_test_scores(rfc, X_test, y_test, experiment=exp)
    """
    assert is_regressor(estimator) or is_classifier(estimator),\
        'Estimator should be sklearn regressor or classifier.'

    exp = _validate_experiment(experiment)

    if y_pred is None:
        y_pred = estimator.predict(X_test)

    if is_regressor(estimator):
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
            r2 = estimator.score(X_test, y_test)
            exp.log_metric('r2_sklearn', r2)
    elif is_classifier(estimator):
        for name, values in zip(['precision', 'recall', 'fbeta_score', 'support'],
                                precision_recall_fscore_support(y_test, y_pred)):
            for i, value in enumerate(values):
                exp.log_metric('{}_class_{}_sklearn'.format(name, i), value)


def _validate_experiment(experiment):
    if experiment is not None:
        if not isinstance(experiment, neptune.experiments.Experiment):
            ValueError('Passed experiment is not Neptune experiment. Create one by using "create_experiment()"')
    else:
        try:
            experiment = neptune.get_experiment()
        except neptune.exceptions.NeptuneNoExperimentContextException:
            raise neptune.exceptions.NeptuneNoExperimentContextException()

    return experiment
