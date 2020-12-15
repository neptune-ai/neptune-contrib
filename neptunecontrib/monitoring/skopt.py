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

import matplotlib.pyplot as plt
import neptune
import numpy as np
import skopt.plots as sk_plots
from skopt.utils import dump

from neptunecontrib.monitoring.utils import axes2fig


class NeptuneCallback:
    """Logs hyperparameter optimization process to Neptune.

    Specifically using NeptuneCallback will log: run metrics and run parameters, best run metrics so far, and
    the current results checkpoint.

    Examples:
        Initialize NeptuneCallback::

            import neptune
            import neptunecontrib.monitoring.skopt as sk_utils

            neptune.init(api_token='ANONYMOUS',
                         project_qualified_name='shared/showroom')

            neptune.create_experiment(name='optuna sweep')

            neptune_callback = sk_utils.NeptuneCallback()

        Run skopt training passing neptune_callback as a callback::

            ...
            results = skopt.forest_minimize(objective, space, callback=[neptune_callback],
                                base_estimator='ET', n_calls=100, n_random_starts=10)

        You can explore an example experiment in Neptune:
        https://ui.neptune.ai/o/shared/org/showroom/e/SHOW-1065/logs
    """

    def __init__(self, experiment=None, log_checkpoint=True):
        self._exp = experiment if experiment else neptune
        self.log_checkpoint = log_checkpoint
        self._iteration = 0

    def __call__(self, res):
        self._exp.log_metric('run_score', x=self._iteration, y=res.func_vals[-1])
        self._exp.log_metric('best_so_far_run_score', x=self._iteration, y=np.min(res.func_vals))
        self._exp.log_text('run_parameters', x=self._iteration, y=NeptuneCallback._get_last_params(res))

        if self.log_checkpoint:
            self._exp.log_artifact(_export_results_object(res), 'results.pkl')
        self._iteration += 1

    @staticmethod
    def _get_last_params(res):
        param_vals = res.x_iters[-1]
        named_params = _format_to_named_params(param_vals, res)
        return str(named_params)


def log_results(results, experiment=None, log_plots=True, log_pickle=True):
    """Logs runs results and parameters to neptune.

    Logs all hyperparameter optimization results to Neptune. Those include best score ('best_score' metric),
    best parameters ('best_parameters' property), convergence plot ('diagnostics' log),
    evaluations plot ('diagnostics' log), and objective plot ('diagnostics' log).

    Args:
        results('scipy.optimize.OptimizeResult'): Results object that is typically an output
          | of the function like `skopt.forest_minimize(...)`
        experiment(`neptune.experiments.Experiment`): Neptune experiment. Default is None.
        log_plots: ('bool'): If True skopt plots will be logged to Neptune.
        log_pickle: ('bool'): if True pickled skopt results object will be logged to Neptune.

    Examples:
        Run skopt training::

            ...
            results = skopt.forest_minimize(objective, space,
                                            base_estimator='ET', n_calls=100, n_random_starts=10)

        Initialize Neptune::

            import neptune

            neptune.init(api_token='ANONYMOUS',
                         project_qualified_name='shared/showroom')
            neptune.create_experiment(name='optuna sweep')

        Send best parameters to Neptune::

            import neptunecontrib.monitoring.skopt as sk_utils

            sk_utils.log_results(results)

    You can explore an example experiment in Neptune:
    https://ui.neptune.ai/o/shared/org/showroom/e/SHOW-1065/logs
    """
    _exp = experiment if experiment else neptune

    _log_best_score(results, _exp)
    _log_best_parameters(results, _exp)

    if log_plots:
        _log_plot_convergence(results, _exp)
        _log_plot_evaluations(results, _exp)
        _log_plot_regret(results, _exp)
        _log_plot_objective(results, _exp)

    if log_pickle:
        _log_results_object(results, _exp)


def NeptuneMonitor(*args, **kwargs):
    message = """NeptuneMonitor was renamed to NeptuneCallback and will be removed in future releases.
    """
    warnings.warn(message)
    return NeptuneCallback(*args, **kwargs)


def _log_best_parameters(results, experiment):
    named_params = ([(dimension.name, param) for dimension, param in zip(results.space, results.x)])
    experiment.set_property('best_parameters', str(named_params))


def _log_best_score(results, experiment):
    experiment.log_metric('best_score', results.fun)


def _log_plot_convergence(results, experiment, name='diagnostics'):
    fig, ax = plt.subplots()
    sk_plots.plot_convergence(results, ax=ax)
    experiment.log_image(name, fig)


def _log_plot_regret(results, experiment, name='diagnostics'):
    fig, ax = plt.subplots()
    sk_plots.plot_regret(results, ax=ax)
    experiment.log_image(name, fig)


def _log_plot_evaluations(results, experiment, name='diagnostics'):
    fig = plt.figure(figsize=(16, 12))
    fig = axes2fig(sk_plots.plot_evaluations(results, bins=10), fig=fig)
    experiment.log_image(name, fig)


def _log_plot_objective(results, experiment, name='diagnostics'):
    try:
        fig = plt.figure(figsize=(16, 12))
        fig = axes2fig(sk_plots.plot_objective(results), fig=fig)
        experiment.log_image(name, fig)
    except Exception as e:
        print('Could not create the objective chart due to error: {}'.format(e))


def _log_results_object(results, experiment=None):
    experiment.log_artifact(_export_results_object(results), 'results.pkl')


def _export_results_object(results):
    from io import BytesIO

    results.specs['args'].pop('callback', None)

    buffer = BytesIO()
    dump(results, buffer, store_objective=False)
    buffer.seek(0)

    return buffer


def _format_to_named_params(params, result):
    return [(dimension.name, param) for dimension, param in zip(result.space, params)]
