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
from sklearn.base import is_regressor
from sklearn.metrics import explained_variance_score, max_error, mean_absolute_error, r2_score
from sklearn.utils import estimator_html_repr

from neptunecontrib.api.html import log_html
from neptunecontrib.api.table import log_table
from neptunecontrib.api.utils import log_pickle


def log_regressor_summary(regressor,
                          X_train=None,
                          X_test=None,
                          y_train=None,
                          y_test=None,
                          experiment=None,
                          log_params=True,
                          log_model=True,
                          log_test_preds=True,
                          log_visualizations=True,
                          log_test_metrics=True):
    """
    Log sklearn regressor summary
    """
    y_pred = None

    if log_test_preds or log_test_metrics:
        try:
            y_pred = regressor.predict(X_test)
        except ValueError:
            print('cannot run "predict" on regressor. Will not log test predictions and test metrics')
            log_test_preds = False
            log_test_metrics = False

    if experiment:
        _exp = experiment
    else:
        try:
            neptune.get_experiment()
            _exp = neptune
        except neptune.exceptions.NeptuneNoExperimentContextException:
            raise neptune.exceptions.NeptuneNoExperimentContextException()

    if not is_regressor(regressor):
        raise ValueError('"regressor" is not sklearn regressor. This method works only with sklearn regressors')

    if log_params:
        try:
            for param, value in regressor.get_params().items():
                _exp.set_property(param, value)
        except Exception:
            print('Did not log params')

    if log_model:
        try:
            log_pickle('model/regressor.skl', regressor, _exp)
        except Exception:
            print('Did not log pickled model')

    if log_test_preds:
        try:
            if len(y_pred.shape) == 1:
                df = pd.DataFrame(data={'y_true': y_test, 'y_pred': y_pred})
                log_table('test_predictions', df, _exp)
        except Exception:
            print('Did not log predictions as table')

        # multioutput regression
        try:
            if len(y_pred.shape) == 2:
                df = pd.DataFrame()
                for j in range(y_pred.shape[1]):
                    df['y_test_output_{}'.format(j)] = y_test[:, j]
                    df['y_pred_output_{}'.format(j)] = y_pred[:, j]
                log_table('test_predictions', df, _exp)
        except Exception:
            print('Did not log predictions as table')

    if log_visualizations:
        try:
            fig, ax = plt.subplots()
            skplt.estimators.plot_learning_curve(regressor, X_train, y_train, ax=ax)
            _exp.log_image('sklearn_charts', fig, image_name='Learning Curve')
        except Exception:
            print('Did not log learning curve chart')

        try:
            fig, ax = plt.subplots()
            skplt.estimators.plot_feature_importances(regressor, ax=ax)
            _exp.log_image('sklearn_charts', fig, image_name='Feature Importance')
        except Exception:
            print('Did not log feature importance chart')

        try:
            log_html('estimator_visualization', estimator_html_repr(regressor), _exp)
        except Exception:
            print('Did not log estimator visualization as html')

    if log_test_metrics:
        try:
            evs = explained_variance_score(y_test, y_pred)
            me = max_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            _exp.log_metric('evs_sklearn', evs)
            _exp.log_metric('me_sklearn', me)
            _exp.log_metric('mae_sklearn', mae)
            _exp.log_metric('r2_sklearn', r2)
        except Exception:
            print('Did not log test metrics')
