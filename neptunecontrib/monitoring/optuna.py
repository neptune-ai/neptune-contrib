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
import warnings

import neptune

from neptunecontrib.api import log_chart, pickle_and_log_artifact


class NeptuneCallback:
    """Logs hyperparameter optimization process to Neptune.

    For each iteration it logs run metric and run parameters as well as the best score to date.

    Args:
        experiment(`neptune.experiments.Experiment`): Neptune experiment. Default is None.
        log_study('bool'): Whether optuna study object should be pickled and logged. By default it is not.
        log_charts('bool'): Deprecated argument. Whether all optuna visualizations plots should be logged.
            By default they are not.
            To log all the charts set log_charts=True.
            If you want to log a particular chart change the argument for that chart explicitly.
            For example log_charts=False and log_slice=True will log only the slice plot to Neptune.
        log_optimization_history('bool'): Whether optuna optimization history chart should be logged.
           By default it is not.
        log_contour('bool'): Whether optuna contour plot should be logged. By default it is not.
        log_parallel_coordinate('bool'): Whether optuna parallel coordinate plot should be logged. By default it is not.
        log_slice('bool'): Whether optuna slice chart should be logged. By default it is not.
        params(`list`): List of parameters to be visualized. Default is all parameters.

    Examples:
        Initialize neptune_monitor::

            import neptune
            import neptunecontrib.monitoring.optuna as opt_utils

            neptune.init(api_token='ANONYMOUS',
                         project_qualified_name='shared/showroom')
            neptune.create_experiment(name='optuna sweep')

            neptune_callback = opt_utils.NeptuneCallback()

        Run Optuna training passing neptune_callback as callback::

            ...
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=100, callbacks=[neptune_callback])

        You can explore an example experiment in Neptune:
        https://ui.neptune.ai/o/shared/org/showroom/e/SHOW-1016/artifacts

        You can also log optuna visualization charts and study object after every iteration::

            neptune_callback = opt_utils.NeptuneCallback(log_charts=True, log_study=True)
    """

    def __init__(self, experiment=None,
                 log_study=False,
                 log_charts=False,
                 log_optimization_history=False,
                 log_contour=False,
                 log_parallel_coordinate=False,
                 log_slice=False,
                 params=None):  # pylint: disable=W0621
        self.exp = experiment if experiment else neptune
        self.log_study = log_study

        if log_charts:

            message = """log_charts argument is depraceted and will be removed in future releases.
            Please use log_optimization_history, log_contour, log_parallel_coordinate, log_slice, arguments explicitly.
            """
            warnings.warn(message)

            log_optimization_history = True
            log_contour = True
            log_parallel_coordinate = True
            log_slice = True

        self.log_optimization_history = log_optimization_history
        self.log_contour = log_contour
        self.log_parallel_coordinate = log_parallel_coordinate
        self.log_slice = log_slice
        self.params = params

    def __call__(self, study, trial):
        import optuna.visualization as vis

        self.exp.log_metric('run_score', trial.value)
        self.exp.log_metric('best_so_far_run_score', study.best_value)
        self.exp.log_text('run_parameters', str(trial.params))

        if self.log_study:
            pickle_and_log_artifact(study, 'study.pkl', experiment=self.exp)

        if self.log_optimization_history:
            log_chart(name='optimization_history', chart=vis.plot_optimization_history(study), experiment=self.exp)
        if self.log_contour:
            log_chart(name='contour', chart=vis.plot_contour(study, params=self.params), experiment=self.exp)
        if self.log_parallel_coordinate:
            log_chart(name='parallel_coordinate', chart=vis.plot_parallel_coordinate(study, params=self.params),
                      experiment=self.exp)
        if self.log_slice:
            log_chart(name='slice', chart=vis.plot_slice(study, params=self.params), experiment=self.exp)


def log_study_info(study, experiment=None,
                   log_study=True,
                   log_charts=True,
                   log_optimization_history=False,
                   log_contour=False,
                   log_parallel_coordinate=False,
                   log_slice=False,
                   params=None):
    """Logs runs results and parameters to neptune.

    Logs all hyperparameter optimization results to Neptune. Those include best score ('best_score' metric),
    best parameters ('best_parameters' property), the study object itself as artifact, and interactive optuna charts
    ('contour', 'parallel_coordinate', 'slice', 'optimization_history') as artifacts in 'charts' sub folder.

    Args:
        study('optuna.study.Study'): Optuna study object after training is completed.
        experiment(`neptune.experiments.Experiment`): Neptune experiment. Default is None.
        log_study('bool'): Whether optuna study object should be logged as pickle. Default is True.
        log_charts('bool'): Deprecated argument. Whether all optuna visualizations charts should be logged.
            By default all charts are sent.
            To not log any charts set log_charts=False.
            If you want to log a particular chart change the argument for that chart explicitly.
            For example log_charts=False and log_slice=True will log only the slice plot to Neptune.
        log_optimization_history('bool'): Whether optuna optimization history chart should be logged. Default is True.
        log_contour('bool'): Whether optuna contour plot should be logged. Default is True.
        log_parallel_coordinate('bool'): Whether optuna parallel coordinate plot should be logged. Default is True.
        log_slice('bool'): Whether optuna slice chart should be logged. Default is True.
        params(`list`): List of parameters to be visualized. Default is all parameters.

    Examples:
        Initialize neptune_monitor::

            import neptune
            import neptunecontrib.monitoring.optuna as opt_utils

            neptune.init(project_qualified_name='USER_NAME/PROJECT_NAME')
            neptune.create_experiment(name='optuna sweep')

            neptune_callback = opt_utils.NeptuneCallback()

        Run Optuna training passing monitor as callback::

            ...
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=100, callbacks=[neptune_callback])
            opt_utils.log_study_info(study)

        You can explore an example experiment in Neptune:
        https://ui.neptune.ai/o/shared/org/showroom/e/SHOW-1016/artifacts
     """
    import optuna.visualization as vis

    _exp = experiment if experiment else neptune

    _exp.log_metric('best_score', study.best_value)
    _exp.set_property('best_parameters', study.best_params)

    if log_charts:
        message = """log_charts argument is depraceted and will be removed in future releases.
        Please use log_optimization_history, log_contour, log_parallel_coordinate, log_slice, arguments explicitly.
        """
        warnings.warn(message)

        log_optimization_history = True
        log_contour = True
        log_parallel_coordinate = True
        log_slice = True

    if log_study:
        pickle_and_log_artifact(study, 'study.pkl', experiment=_exp)
    if log_optimization_history:
        log_chart(name='optimization_history', chart=vis.plot_optimization_history(study), experiment=_exp)
    if log_contour:
        log_chart(name='contour', chart=vis.plot_contour(study, params=params), experiment=_exp)
    if log_parallel_coordinate:
        log_chart(name='parallel_coordinate', chart=vis.plot_parallel_coordinate(study, params=params), experiment=_exp)
    if log_slice:
        log_chart(name='slice', chart=vis.plot_slice(study, params=params), experiment=_exp)
