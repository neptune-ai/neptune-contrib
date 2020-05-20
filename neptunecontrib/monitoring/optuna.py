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


def NeptuneCallback(experiment=None, log_charts=False, params=None):
    """Logs hyperparameter optimization process to Neptune.

    For each iteration it logs run metric and run parameters as well as the best score to date.

    Args:
        experiment(`neptune.experiments.Experiment`): Neptune experiment. Default is None.
        log_charts('bool'): Whether optuna visualization charts should be logged. By default no charts are logged.
        params(`list`): List of parameters to be visualized. Default is all parameters.

    Examples:
        Initialize neptune_monitor::

            import neptune
            import neptunecontrib.monitoring.optuna as opt_utils

            neptune.init(api_token='ANONYMOUS',
                         project_qualified_name='shared/showroom')
            neptune.create_experiment(name='optuna sweep')

            neptune_callback = opt_utils.NeptuneCallback()

        Run Optuna training passing monitor as callback::

            ...
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=100, callbacks=[neptune_callback])

        You can explore an example experiment in Neptune:
        https://ui.neptune.ai/o/shared/org/showroom/e/SHOW-1016/artifacts

        You can also log optuna visualization charts after every iteration::

            neptune_callback = opt_utils.NeptuneCallback(log_charts=True)
    """
    import optuna.visualization as vis

    _exp = experiment if experiment else neptune

    def monitor(study, trial):
        _exp.log_metric('run_score', trial.value)
        _exp.log_metric('best_so_far_run_score', study.best_value)
        _exp.log_text('run_parameters', str(trial.params))

        if log_charts:
            log_chart(
                name='optimization_history', chart=vis.plot_optimization_history(study), experiment=_exp)
            log_chart(
                name='contour', chart=vis.plot_contour(study, params=params), experiment=_exp)
            log_chart(
                name='parallel_coordinate', chart=vis.plot_parallel_coordinate(study, params=params), experiment=_exp)
            log_chart(
                name='slice', chart=vis.plot_slice(study, params=params), experiment=_exp)

    return monitor


def log_study(study, experiment=None, log_charts=True, params=None):
    """Logs runs results and parameters to neptune.

    Logs all hyperparameter optimization results to Neptune. Those include best score ('best_score' channel),
    best parameters ('best_parameters' property), the study object itself, and interactive optuna charts
    ('contour', 'parallel_coordinate', 'slice', 'optimization_history').

    Args:
        results('optuna.study.Study'): Optuna study object after training is completed.
        experiment(`neptune.experiments.Experiment`): Neptune experiment. Default is None.
        log_charts('bool'): Whether optuna visualization charts should be logged. By default all charts are logged.
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
            opt_utils.log_study(study)

        You can explore an example experiment in Neptune:
        https://ui.neptune.ai/o/shared/org/showroom/e/SHOW-1016/artifacts
     """
    import optuna.visualization as vis

    _exp = experiment if experiment else neptune

    _exp.log_metric('best_score', study.best_value)
    _exp.set_property('best_parameters', study.best_params)

    if log_charts:
        log_chart(name='optimization_history', chart=vis.plot_optimization_history(study), experiment=_exp)
        log_chart(name='contour', chart=vis.plot_contour(study, params=params), experiment=_exp)
        log_chart(name='parallel_coordinate', chart=vis.plot_parallel_coordinate(study, params=params), experiment=_exp)
        log_chart(name='slice', chart=vis.plot_slice(study, params=params), experiment=_exp)

    pickle_and_log_artifact(study, 'study.pkl', experiment=_exp)


def NeptuneMonitor(experiment=None):
    message = """NeptuneMonitor was renamed to NeptuneCallback and will be removed in future releases.
    """
    warnings.warn(message)
    return NeptuneCallback(experiment=experiment)
