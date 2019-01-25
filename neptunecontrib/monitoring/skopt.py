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
import skopt.plots as sk_plots

from neptunecontrib.monitoring.utils import fig2pil
from neptunecontrib.viz.utils import axes2fig


class NeptuneMonitor:
    """Logs hyperparameter optimization process to Neptune.

    Examples:
        Initialize NeptuneMonitor.

        >>> import neptune
        >>> import neptunecontrib.monitoring.skopt as sk_utils
        >>> ctx = neptune.Context()
        >>> monitor = sk_utils.NeptuneMonitor(ctx)

        Run skopt training passing monitor as a a callback

        >>> ...
        >>> results = skopt.forest_minimize(objective, space, callback=[monitor],
                                base_estimator='ET', n_calls=100, n_random_starts=10)

    """

    def __init__(self, ctx):
        self._ctx = ctx
        self._iteration = 0

    def __call__(self, res):
        self._ctx.channel_send('hyperparameter_search_score',
                               x=self._iteration, y=res.func_vals[-1])
        self._ctx.channel_send('search_parameters',
                               x=self._iteration, y=NeptuneMonitor._get_last_params(res))
        self._iteration += 1

    @staticmethod
    def _get_last_params(res):
        param_vals = res.x_iters[-1]
        named_params = _format_to_named_params(param_vals, res)
        return named_params


def send_runs(results, ctx):
    """Logs runs results and parameters to neptune.

    Text channel `hyperparameter_search_score` is created and a list of tuples (name, value)
    of best paramters is logged to neptune.

    Args:
        results('scipy.optimize.OptimizeResult'): Results object that is typically an
            output of the function like `skopt.forest_minimize(...)`
        ctx(`neptune.Context`): Neptune context.

    Examples:
        Run skopt training.

        >>> results = skopt.forest_minimize(objective, space,
                                base_estimator='ET', n_calls=100, n_random_starts=10)

        Send best parameters to neptune.

        >>> import neptune
        >>> import neptunecontrib.monitoring.skopt as sk_utils
        >>> ctx = neptune.Context()
        >>> sk_monitor.send_best_parameters(results, ctx)

    """
    for loss, params in zip(results.func_vals, results.x_iters):
        ctx.channel_send('hyperparameter_search_score', y=loss)

        named_params = _format_to_named_params(params, results)
        ctx.channel_send('search_parameters', named_params)


def send_best_parameters(results, ctx):
    """Logs best_parameters list to neptune.

    Text channel `best_parameters` is created and a list of tuples (name, value)
    of best paramters is logged to neptune.

    Args:
        results('scipy.optimize.OptimizeResult'): Results object that is typically an
            output of the function like `skopt.forest_minimize(...)`
        ctx(`neptune.Context`): Neptune context.

    Examples:
        Run skopt training.

        >>> results = skopt.forest_minimize(objective, space,
                                base_estimator='ET', n_calls=100, n_random_starts=10)

        Send best parameters to neptune.

        >>> import neptune
        >>> import neptunecontrib.monitoring.skopt as sk_utils
        >>> ctx = neptune.Context()
        >>> sk_monitor.send_best_parameters(results, ctx)

    """
    param_vals = results.x
    named_params = _format_to_named_params(param_vals, results)
    ctx.channel_send('best_parameters', named_params)


def send_plot_convergence(results, ctx):
    """Logs skopt plot_convergence figure to neptune.

    Image channel `convergence` is created and the output of the
    plot_convergence function is first covented to `neptune.Image` and
    then sent to neptune.

    Args:
        results('scipy.optimize.OptimizeResult'): Results object that is typically an
            output of the function like `skopt.forest_minimize(...)`
        ctx(`neptune.Context`): Neptune context.

    Examples:
        Run skopt training.

        >>> results = skopt.forest_minimize(objective, space,
                                base_estimator='ET', n_calls=100, n_random_starts=10)

        Send skopt plot_convergence figure to neptune.

        >>> import neptune
        >>> import neptunecontrib.monitoring.skopt as sk_utils
        >>> ctx = neptune.Context()
        >>> sk_monitor.send_plot_convergence(results, ctx)

    """
    convergence = fig2pil(axes2fig(sk_plots.plot_convergence(results)))
    ctx.channel_send('convergence', neptune.Image(
        name='convergence',
        description="plot_convergence from skopt",
        data=convergence))


def send_plot_evaluations(results, ctx):
    """Logs skopt plot_evaluations figure to neptune.

    Image channel `evaluations` is created and the output of the
    plot_evaluations function is first covented to `neptune.Image` and
    then sent to neptune.

    Args:
        results('scipy.optimize.OptimizeResult'): Results object that is typically an
            output of the function like `skopt.forest_minimize(...)`
        ctx(`neptune.Context`): Neptune context.

    Examples:
        Run skopt training.

        >>> results = skopt.forest_minimize(objective, space,
                                base_estimator='ET', n_calls=100, n_random_starts=10)

        Send skopt plot_evaluations figure to neptune.

        >>> import neptune
        >>> import neptunecontrib.monitoring.skopt as sk_utils
        >>> ctx = neptune.Context()
        >>> sk_monitor.send_plot_evaluations(results, ctx)

    """
    evaluations = fig2pil(axes2fig(sk_plots.plot_evaluations(results, bins=10)))
    ctx.channel_send('evaluations', neptune.Image(
        name='evaluations',
        description="plot_evaluations from skopt",
        data=evaluations))


def send_plot_objective(results, ctx):
    """Logs skopt plot_objective figure to neptune.

    Image channel `objective` is created and the output of the
    plot_objective function is first covented to `neptune.Image` and
    then sent to neptune.

    Args:
        results('scipy.optimize.OptimizeResult'): Results object that is typically an
            output of the function like `skopt.forest_minimize(...)`
        ctx(`neptune.Context`): Neptune context.

    Examples:
        Run skopt training.

        >>> results = skopt.forest_minimize(objective, space,
                                base_estimator='ET', n_calls=100, n_random_starts=10)

        Send skopt plot_objective figure to neptune.

        >>> import neptune
        >>> import neptunecontrib.monitoring.skopt as sk_utils
        >>> ctx = neptune.Context()
        >>> sk_monitor.send_plot_objective(results, ctx)

    """
    try:
        objective = fig2pil(axes2fig(sk_plots.plot_objective(results)))
        ctx.channel_send('objective', neptune.Image(
            name='objective',
            description="plot_objective from skopt",
            data=objective))
    except Exception as e:
        print('Could not create ans objective chart due to error: {}'.format(e))


def _format_to_named_params(params, result):
    param_names = [dim.name for dim in result.space.dimensions]
    named_params = []
    for name, val in zip(param_names, params):
        named_params.append((name, val))
    return named_params
