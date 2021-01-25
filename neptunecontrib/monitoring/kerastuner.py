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

from kerastuner.engine.logger import Logger


class NeptuneLogger(Logger):
    """Logs hyperparameter optimization process to Neptune.

    For each iteration it logs run parameters ('hyperparameters/values' text log),
     and all the metrics and losses to Neptune.

    Args:
        experiment(`neptune.experiments.Experiment`): Neptune experiment. Default is None.

    Examples:
        Initialize neptune_monitor::

            import neptune
            import neptunecontrib.monitoring.kerastuner as npt_utils

            neptune.init(api_token='ANONYMOUS',
                         project_qualified_name='shared/keras-tuner-integration')
            neptune.create_experiment(name='bayesian-sweep')

            neptune_logger = npt_utils.NeptuneLogger()

        Run Keras Tuner search passing neptune_logger as logger::

            ...
            tuner =  BayesianOptimization(
                build_model,
                objective='val_accuracy',
                max_trials=10,
                num_initial_points=3,
                executions_per_trial=3,
                project_name='bayesian-sweep',
                logger=npt_utils.NeptuneLogger())

        You can explore an example experiment in Neptune:
        https://ui.neptune.ai/o/shared/org/keras-tuner-integration/e/KER-19/charts
    """

    def __init__(self, experiment=None):
        self.exp = experiment if experiment else neptune

    def report_trial_state(self, trial_id, trial_state):
        """Gives the logger information about trial status."""

        self.exp.log_text('hyperparameters/values', str(trial_state['hyperparameters']['values']))

        for name, vals in trial_state['metrics']['metrics'].items():
            metric_values = vals['observations'][0]['value']
            avg_metric_value = sum(metric_values) / len(metric_values)
            self.exp.log_metric(name, avg_metric_value)

    def register_tuner(self, tuner_state):
        pass

    def register_trial(self, trial_id, trial_state):
        pass

    def exit(self):
        pass


def log_tuner_info(tuner, experiment=None, log_project_dir=True):
    """Logs runs results and parameters to neptune.

    Logs all hyperparameter optimization results to Neptune. Those include best score ('best_score' metric),
    best parameters ('best_parameters' property), score for every run ('run_score' metric),
    tuner project directory as an artifact, parameter space ('hyperparameters/space' text log),
    tuner id ('tuner_id' property), best trial id ('best_trial_id' property),
    name of the metric/loss used as objective, and it's direction ('objective/name' and 'objective/direction' property).

    Args:
        tuner('kerastuner.engine.tuner.Tuner'): Keras Tuner object after training is completed.
        experiment(`neptune.experiments.Experiment`): Neptune experiment. Default is None.
        log_project_dir('bool'): Whether Keras Tuner project directory, with all the trial information,
            should be logged to Neptune.

    Examples:
        Initialize neptune experiment::

            import neptune

            neptune.init(api_token='ANONYMOUS',
                         project_qualified_name='shared/keras-tuner-integration')
            neptune.create_experiment(name='bayesian-sweep')

        Run Keras Tuner search::

            ...
            tuner =  BayesianOptimization(...)
            tuner.search(x=x, y=y,
                         epochs=5,
                         validation_data=(val_x, val_y))

        Log information from the Tuner object to Neptune::

            import neptunecontrib.monitoring.kerastuner as npt_utils
            npt_utils.log_tuner_info(tuner)

        You can explore an example experiment in Neptune:
        https://ui.neptune.ai/o/shared/org/keras-tuner-integration/e/KER-19/details
     """
    exp = experiment if experiment else neptune

    if log_project_dir:
        exp.log_artifact(tuner.project_dir)

    exp.set_property('best_parameters', tuner.get_best_hyperparameters()[0].values)
    exp.set_property('objective/name', tuner.oracle.objective.name)
    exp.set_property('objective/direction', tuner.oracle.objective.direction)
    exp.set_property('objective/direction', tuner.oracle.objective.direction)
    exp.set_property('tuner_id', tuner.tuner_id)
    exp.set_property('best_trial_id', tuner.oracle.get_best_trials()[0].trial_id)

    for dim in tuner.oracle.get_space().space:
        exp.log_text('hyperparameters/space', str(dim))

    exp.log_metric('best_score', tuner.oracle.get_best_trials()[0].score)

    for _, trial in tuner.oracle.trials.items():
        exp.log_metric('run_score', trial.score)
