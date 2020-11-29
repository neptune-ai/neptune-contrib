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
import scikitplot as skplt
from sklearn.base import is_regressor, is_classifier
from sklearn.metrics import explained_variance_score, max_error, mean_absolute_error, r2_score
from sklearn.utils import estimator_html_repr
from yellowbrick.regressor import ResidualsPlot, PredictionError, CooksDistance

from neptunecontrib.api.html import log_html
from neptunecontrib.api.table import log_table
from neptunecontrib.api.utils import log_pickle


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
        y_pred_proba = estimator.predict_proba(data)
        return y_pred, y_pred_proba


def _log_estimator_params(flag, estimator, experiment):
    if flag:
        try:
            for param, value in estimator.get_params().items():
                experiment.set_property(param, value)
        except Exception:
            experiment.log_text('sklearn-logging-summary', 'Did not log parameters.')


def _log_pickled_model(flag, estimator, experiment):
    if flag:
        try:
            log_pickle('model/estimator.skl', estimator, experiment)
        except Exception:
            experiment.log_text('sklearn-logging-summary', 'Did not log pickled model.')


def _log_test_predictions(flag, y_pred, y_test, experiment):
    if flag:
        # single output
        if len(y_pred.shape) == 1:
            try:
                df = pd.DataFrame(data={'y_true': y_test, 'y_pred': y_pred})
                log_table('test_predictions', df, experiment)
            except Exception:
                experiment.log_text('sklearn-logging-summary', 'Did not log test predictions as table.')

        # multi output
        if len(y_pred.shape) == 2:
            try:
                df = pd.DataFrame()
                for j in range(y_pred.shape[1]):
                    df['y_test_output_{}'.format(j)] = y_test[:, j]
                    df['y_pred_output_{}'.format(j)] = y_pred[:, j]
                log_table('test_predictions', df, experiment)
            except Exception:
                experiment.log_text('sklearn-logging-summary', 'Did not log test predictions as table.')


def _log_test_predictions_probabilities(flag, classifier, y_pred_proba, experiment):
    if flag:
        try:
            df = pd.DataFrame(data=y_pred_proba, columns=classifier.classes_)
            log_table('test_preds_proba', df, experiment)
        except Exception:
            experiment.log_text('sklearn-logging-summary', 'Did not log test predictions probabilities as table.')


def log_regressor_summary(regressor,
                          X_train=None,
                          X_test=None,
                          y_train=None,
                          y_test=None,
                          experiment=None,
                          log_params=True,
                          log_model=True,
                          log_test_preds=True,
                          log_test_metrics=True,
                          log_visualizations=True):
    """
    Log sklearn regressor summary.

    This method automatically logs all regressor parameters, pickled estimator (model), test predictions as table,
    model performance visualizations, sklearn pipeline as an interactive graph and test metrics.
    """
    exp = _check_experiment(experiment)
    _check_estimator(regressor, 'regressor')

    y_pred = _compute_test_preds(regressor, X_test)

    _log_estimator_params(log_params, regressor, exp)
    _log_pickled_model(log_model, regressor, exp)
    _log_test_predictions(log_test_preds, y_pred, y_test, exp)

    if log_test_metrics:
        # single output
        if len(y_pred.shape) == 1:
            try:
                evs = explained_variance_score(y_test, y_pred)
                me = max_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                exp.log_metric('evs_sklearn', evs)
                exp.log_metric('me_sklearn', me)
                exp.log_metric('mae_sklearn', mae)
                exp.log_metric('r2_sklearn', r2)
            except Exception:
                experiment.log_text('sklearn-logging-summary', 'Did not log test metrics.')

        # multi output
        if len(y_pred.shape) == 2:
            try:
                r2 = regressor.score(X_test, y_test)
                exp.log_metric('r2_sklearn', r2)
            except Exception:
                experiment.log_text('sklearn-logging-summary', 'Did not log test metrics.')

    if log_visualizations:
        try:
            fig, ax = plt.subplots()
            skplt.estimators.plot_learning_curve(regressor, X_train, y_train, ax=ax)
            exp.log_image('sklearn_charts', fig, image_name='Learning Curve')
        except Exception:
            experiment.log_text('sklearn-logging-summary', 'Did not log learning curve chart.')

        try:
            fig, ax = plt.subplots()
            skplt.estimators.plot_feature_importances(regressor, ax=ax)
            exp.log_image('sklearn_charts', fig, image_name='Feature Importance')
        except Exception:
            experiment.log_text('sklearn-logging-summary', 'Did not log feature importance chart.')

        try:
            log_html('estimator_visualization', estimator_html_repr(regressor), exp)
        except Exception:
            experiment.log_text('sklearn-logging-summary', 'Did not log estimator visualization as html.')

        try:
            fig, ax = plt.subplots()
            visualizer = ResidualsPlot(regressor, is_fitted=True, ax=ax)
            visualizer.fit(X_train, y_train)
            visualizer.score(X_test, y_test)
            visualizer.finalize()
            exp.log_image('sklearn_charts', fig, image_name='Residuals Plot')
        except Exception:
            experiment.log_text('sklearn-logging-summary', 'Did not log residuals plot chart.')

        try:
            fig, ax = plt.subplots()
            visualizer = PredictionError(regressor, is_fitted=True, ax=ax)
            visualizer.fit(X_train, y_train)
            visualizer.score(X_test, y_test)
            visualizer.finalize()
            exp.log_image('sklearn_charts', fig, image_name='Prediction Error')
        except Exception:
            experiment.log_text('sklearn-logging-summary', 'Did not log prediction error chart.')

        try:
            fig, ax = plt.subplots()
            visualizer = CooksDistance(ax=ax)
            visualizer.fit(X_train, y_train)
            visualizer.finalize()
            exp.log_image('sklearn_charts', fig, image_name='Cooks Distance')
        except Exception:
            experiment.log_text('sklearn-logging-summary', 'Did not log cooks distance chart.')

        plt.close('all')


def log_classifier_summary(classifier,
                           X_train=None,
                           X_test=None,
                           y_train=None,
                           y_test=None,
                           experiment=None,
                           log_params=True,
                           log_model=True,
                           log_test_preds=True,
                           log_test_metrics=True,
                           log_visualizations=True):
    """
    Log sklearn classifier summary.

    This method automatically logs all classifier parameters, pickled estimator (model), test predictions as table,
    model performance visualizations, sklearn pipeline as an interactive graph and test metrics.
    """
    exp = _check_experiment(experiment)
    _check_estimator(classifier, 'classifier')

    (y_pred, y_pred_proba) = _compute_test_preds(classifier, X_test)

    _log_estimator_params(log_params, classifier, exp)
    _log_pickled_model(log_model, classifier, exp)
    _log_test_predictions(log_test_preds, y_pred, y_test, exp)
    _log_test_predictions_probabilities(log_test_preds, classifier, y_pred_proba, exp)
