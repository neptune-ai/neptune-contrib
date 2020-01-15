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

import neptune
from neptunecontrib.monitoring.utils import pickle_and_send_artifact


def NeptuneMonitor(experiment=None):
    """Logs hyperparameter optimization process to Neptune.

    Args:
        experiment(`neptune.experiments.Experiment`): Neptune experiment. Default is None.

    Examples:
        Initialize neptune_monitor::

            import neptune
            import neptunecontrib.monitoring.optuna as opt_utils

            neptune.init(project_qualified_name='USER_NAME/PROJECT_NAME')
            neptune.create_experiment(name='optuna sweep')

            monitor = opt_utils.NeptuneMonitor()

        Run Optuna training passing monitor as callback::

            ...
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=100, callbacks=[monitor])

        You can explore an example experiment in Neptune https://ui.neptune.ai/jakub-czakon/blog-hpo/e/BLOG-404/charts
    """

    _exp = experiment if experiment else neptune

    def monitor(study, trial):
        _exp.log_metric('run_score', trial.value)
        _exp.log_metric('best_so_far_run_score', study.best_value)
        _exp.log_text('run_parameters', str(trial.params))

    return monitor


def log_study(study, experiment=None):
    """Logs runs results and parameters to neptune.
    Logs all hyperparameter optimization results to Neptune. Those include best score ('best_score' channel),
    best parameters ('best_parameters' property), and the study object itself.

    Args:
        results('optuna.study.Study'): Optuna study object after training is completed.
        experiment(`neptune.experiments.Experiment`): Neptune experiment. Default is None.

    Examples:
        Initialize neptune_monitor::

            import neptune
            import neptunecontrib.monitoring.optuna as opt_utils

            neptune.init(project_qualified_name='USER_NAME/PROJECT_NAME')
            neptune.create_experiment(name='optuna sweep')

            monitor = opt_utils.NeptuneMonitor()

        Run Optuna training passing monitor as callback::

            ...
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=100, callbacks=[monitor])
            opt_utils.log_study(study)

        You can explore an example experiment in Neptune https://ui.neptune.ai/jakub-czakon/blog-hpo/e/BLOG-404/charts
     """
    _exp = experiment if experiment else neptune

    _exp.log_metric('best_score', study.best_value)
    _exp.set_property('best_parameters', study.best_params)
    pickle_and_send_artifact(study, 'study.pkl', experiment=_exp)
