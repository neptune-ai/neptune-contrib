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

import tempfile

import matplotlib.pyplot as plt
import neptune
import skopt.plots as sk_plots

from neptunecontrib.monitoring.utils import axes2fig


class NeptuneMonitor:
    """Logs hyperparameter optimization process to Neptune.

    Examples:
        Initialize NeptuneMonitor::

            import neptune
            import neptunecontrib.monitoring.skopt as sk_utils

            neptune.init(project_qualified_name='USER_NAME/PROJECT_NAME')

            monitor = sk_utils.NeptuneMonitor()

        Run skopt training passing monitor as a a callback::

            ...
            results = skopt.forest_minimize(objective, space, callback=[monitor],
                                base_estimator='ET', n_calls=100, n_random_starts=10)

    """

    def __init__(self, experiment=None):
        self._exp = experiment if experiment else neptune
        self._iteration = 0

    def __call__(self, res):
        self._exp.send_metric('run_score',
                              x=self._iteration, y=res.func_vals[-1])
        self._exp.send_text('run_parameters',
                            x=self._iteration, y=NeptuneMonitor._get_last_params(res))
        self._iteration += 1

    @staticmethod
    def _get_last_params(res):
        param_vals = res.x_iters[-1]
        named_params = _format_to_named_params(param_vals, res)
        return str(named_params)


def send_runs(results, experiment=None):
    """Logs runs results and parameters to neptune.

    Text channel `hyperparameter_search_score` is created and a list of tuples (name, value)
    of best paramters is logged to neptune.

    Args:
        results('scipy.optimize.OptimizeResult'): Results object that is typically an
            output of the function like `skopt.forest_minimize(...)`
        experiment(`neptune.experiments.Experiment`): Neptune experiment. Default is None.

    Examples:
        Run skopt training::

            ...
            results = skopt.forest_minimize(objective, space,
                                base_estimator='ET', n_calls=100, n_random_starts=10)

        Send best parameters to neptune::

            import neptune
            import neptunecontrib.monitoring.skopt as sk_utils

            neptune.init(project_qualified_name='USER_NAME/PROJECT_NAME')

            sk_monitor.send_best_parameters(results)

    """

    _exp = experiment if experiment else neptune

    for i, (loss, params) in enumerate(zip(results.func_vals, results.x_iters)):
        _exp.send_metric('run_score', x=i, y=loss)

        named_params = _format_to_named_params(params, results)
        _exp.send_text('run_parameters', str(named_params))


def send_best_parameters(results, experiment=None):
    """Logs best_parameters list to neptune.

    Text channel `best_parameters` is created and a list of tuples (name, value)
    of best paramters is logged to neptune.

    Args:
        results('scipy.optimize.OptimizeResult'): Results object that is typically an
            output of the function like `skopt.forest_minimize(...)`
        experiment(`neptune.experiments.Experiment`): Neptune experiment. Default is None.

    Examples:
        Run skopt training::

            ...
            results = skopt.forest_minimize(objective, space,
                                base_estimator='ET', n_calls=100, n_random_starts=10)

        Send best parameters to neptune::

            import neptune
            import neptunecontrib.monitoring.skopt as sk_utils

            neptune.init(project_qualified_name='USER_NAME/PROJECT_NAME')

            sk_monitor.send_best_parameters(results)

    """
    _exp = experiment if experiment else neptune

    named_params = _format_to_named_params(results.x, results)
    _exp.set_property('best_parameters', str(named_params))


def send_plot_convergence(results, experiment=None, channel_name='convergence'):
    """Logs skopt plot_convergence figure to neptune.

    Image channel `convergence` is created and the output of the
    plot_convergence function is first covented to `neptune.Image` and
    then sent to neptune.

    Args:
        results('scipy.optimize.OptimizeResult'): Results object that is typically an
            output of the function like `skopt.forest_minimize(...)`
        experiment(`neptune.experiments.Experiment`): Neptune experiment. Default is None.

    Examples:
        Run skopt training::

            ...
            results = skopt.forest_minimize(objective, space,
                                base_estimator='ET', n_calls=100, n_random_starts=10)

        Send skopt plot_convergence figure to neptune::

            import neptune
            import neptunecontrib.monitoring.skopt as sk_utils

            neptune.init(project_qualified_name='USER_NAME/PROJECT_NAME')

            sk_monitor.send_plot_convergence(results)

    """

    _exp = experiment if experiment else neptune

    fig, ax = plt.subplots(figsize=(16, 12))
    sk_plots.plot_convergence(results, ax=ax)

    with tempfile.NamedTemporaryFile(suffix='.png') as f:
        fig.savefig(f.name)
        _exp.send_image(channel_name, f.name)


def send_plot_evaluations(results, experiment=None, channel_name='evaluations'):
    """Logs skopt plot_evaluations figure to neptune.

    Image channel `evaluations` is created and the output of the
    plot_evaluations function is first covented to `neptune.Image` and
    then sent to neptune.

    Args:
        results('scipy.optimize.OptimizeResult'): Results object that is typically an
            output of the function like `skopt.forest_minimize(...)`
        experiment(`neptune.experiments.Experiment`): Neptune experiment. Default is None.

    Examples:
        Run skopt training::

            ...
            results = skopt.forest_minimize(objective, space,
                                base_estimator='ET', n_calls=100, n_random_starts=10)

        Send skopt plot_evaluations figure to neptune::

            import neptune
            import neptunecontrib.monitoring.skopt as sk_utils

            neptune.init(project_qualified_name='USER_NAME/PROJECT_NAME')

            sk_monitor.send_plot_evaluations(results)

    """
    _exp = experiment if experiment else neptune

    fig = plt.figure(figsize=(16, 12))
    fig = axes2fig(sk_plots.plot_evaluations(results, bins=10), fig=fig)

    with tempfile.NamedTemporaryFile(suffix='.png') as f:
        fig.savefig(f.name)
        _exp.send_image(channel_name, f.name)


def send_plot_objective(results, experiment=None, channel_name='objective'):
    """Logs skopt plot_objective figure to neptune.

    Image channel `objective` is created and the output of the
    plot_objective function is first covented to `neptune.Image` and
    then sent to neptune.

    Args:
        results('scipy.optimize.OptimizeResult'): Results object that is typically an
            output of the function like `skopt.forest_minimize(...)`
        experiment(`neptune.experiments.Experiment`): Neptune experiment. Default is None.

    Examples:
        Run skopt training::

            ...
            results = skopt.forest_minimize(objective, space,
                                            base_estimator='ET', n_calls=100, n_random_starts=10)

        Send skopt plot_objective figure to neptune::

            import neptune
            import neptunecontrib.monitoring.skopt as sk_utils

            neptune.init(project_qualified_name='USER_NAME/PROJECT_NAME')

            sk_monitor.send_plot_objective(results)

    """

    _exp = experiment if experiment else neptune
    fig = plt.figure(figsize=(16, 12))

    try:
        fig = axes2fig(sk_plots.plot_objective(results), fig=fig)
        with tempfile.NamedTemporaryFile(suffix='.png') as f:
            fig.savefig(f.name)
            _exp.send_image(channel_name, f.name)
    except Exception as e:
        print('Could not create ans objective chart due to error: {}'.format(e))


def _format_to_named_params(params, result):
    return ([(dimension.name, param) for dimension, param in zip(result.space, params)])
