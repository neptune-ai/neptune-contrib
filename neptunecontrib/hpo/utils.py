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

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeResult
import skopt


def hyperopt2skopt(trials, space):
    """Converts hyperopt trials to scipy OptimizeResult.

    Helper function that converts the hyperopt Trials instance into scipy OptimizeResult
    format.

    Args:
        trials(`hyperopt.base.Trials`): hyperopt trials object which stores training
            information from the fmin() optimization function.
        space(`collections.OrderedDict`): Hyper parameter space over which
            hyperopt will search. It is important to have this as OrderedDict rather
            than a simple dictionary because otherwise the parameter names will be
            shuffled.

    Returns:
        `scipy.optimize.optimize.OptimizeResult`: Converted OptimizeResult.


    Examples:
        Prepare the space of hyperparameters to search over.

        >>> from hyperopt import hp, tpe, fmin, Trials
        >>> space = OrderedDict(num_leaves=hp.choice('num_leaves', range(10, 60, 1)),
                    max_depth=hp.choice('max_depth', range(2, 30, 1)),
                    feature_fraction=hp.uniform('feature_fraction', 0.1, 0.9)
                   )

        Create an objective and run your hyperopt training

        >>> trials = Trials()
        >>> _ = fmin(objective, space, trials=trials, algo=tpe.suggest, max_evals=100)

        Convert trials object to the OptimizeResult object.

        >>> import neptunecontrib.hpo.utils as hp_utils
        >>> results = hp_utils.hyperopt2skopt(trials, space)
    """
    param_names = list(space.keys())
    skopt_space = _convert_space_hop_skopt(space)
    results_ = {}
    for trial in trials.trials:
        trial_params = [trial['misc']['vals'][name][0] for name in param_names]
        results_.setdefault('x_iters', []).append(trial_params)
        results_.setdefault('func_vals', []).append(trial['result']['loss'])
    optimize_results = OptimizeResult()
    optimize_results.x = [trials.argmin[name] for name in param_names]
    optimize_results.x_iters = results_['x_iters']
    optimize_results.fun = trials.best_trial['result']['loss']
    optimize_results.func_vals = results_['func_vals']
    optimize_results.space = skopt_space
    return optimize_results


def df2result(df, metric_col, param_cols, param_types=None):
    """Converts dataframe with metrics and hyperparameters to the OptimizeResults format.

    It is a helper function that lets you use all the tools that expect OptimizeResult object
    like for example scikit-optimize plot_evaluations function.

    Args:
        df(`pandas.DataFrame`): Dataframe containing metric and hyperparameters.
        metric_col(str): Name of the metric column.
        param_cols(list): Names of the hyperparameter columns.
        param_types(list or None): Optional list of hyperparameter column types.
            By default it will treat all the columns as float but you can also pass str
            for categorical channels. E.g param_types=[float, str, float, float]

    Returns:
        `scipy.optimize.OptimizeResult`: Results object that contains the hyperparameter and metric
        information.

    Examples:
        Instantiate a session.

        >>> from neptune.sessions import Session
        >>> session = Session()

        Fetch a project and a list of experiments.

        >>> project = session.get_projects('neptune-ml')['neptune-ml/Home-Credit-Default-Risk']
        >>> leaderboard = project.get_leaderboard(state=['succeeded'], owner=['czakon'])

        Comvert the leaderboard dataframe to the `ResultOptimize` instance taking only the parameters and
        metric that you care about.

        >>> result = df2result(leaderboard,
        metric_col='channel_ROC_AUC',
        param_cols=['parameter_lgbm__max_depth', 'parameter_lgbm__num_leaves', 'parameter_lgbm__min_child_samples'])

    """

    if not param_types:
        param_types = [float for _ in param_cols]

    df = _prep_df(df, param_cols, param_types)
    df = df.sort_values(metric_col, ascending=False)
    param_space = _convert_to_param_space(df, param_cols, param_types)

    results = OptimizeResult()
    results.x_iters = df[param_cols].values
    results.func_vals = df[metric_col].to_list()
    results.x = results.x_iters[0]
    results.fun = results.func_vals[0]
    results.space = param_space
    return results


def optuna2skopt(results):
    """Converts optuna results to scipy OptimizeResult.

    Helper function that converts the optuna Trials instance into scipy OptimizeResult
    format.

    Args:
        results(`pandas.DataFrame`): Dataframe containing scores and hyperparameters.
            It is the output of running study.trials_dataframe().

    Returns:
        `scipy.optimize.optimize.OptimizeResult`: Converted OptimizeResult.

    Examples:
        Run your optuna study.

        >>> study = optuna.create_study()
        >>> study.optimize(objective, n_trials=100)

        Convert trials_dataframe object to the OptimizeResult object.

        >>> import neptunecontrib.hpo.utils as hp_utils
        >>> results = hp_utils.optuna2skopt(study.trials_dataframe())
    """

    results_ = results['params']
    results_['target'] = results['value']
    return df2result(results_,
                     metric_col='target',
                     param_cols=[col for col in results_.columns if col != 'target'])


def bayes2skopt(results):
    """Converts BayesOptimization results to scipy OptimizeResult.

    Helper function that converts the optuna Trials instance into scipy OptimizeResult
    format.

    Args:
        results(`pandas.DataFrame`): Dataframe containing scores and hyperparameters.
            It is the output of running study.trials_dataframe().

    Returns:
        `scipy.optimize.optimize.OptimizeResult`: Converted OptimizeResult.

    Examples:
        Run BayesOptimize maximization.

        >>> bayes_optimization = BayesianOptimization(objective, space)
        >>> bayes_optimization.maximize(init_points=10, n_iter=100, xi=0.06)

        Convert bayes.space.res() object to the OptimizeResult object.

        >>> import neptunecontrib.hpo.utils as hp_utils
        >>> results = hp_utils.bayes2skopt(bayes_optimization.space.res())
    """

    results = [{'target': trial['target'], **trial['params']} for trial in results]
    results_df = pd.DataFrame(results)
    return df2result(results_df,
                     metric_col='target',
                     param_cols=[col for col in results_df.columns if col != 'target'])


def _prep_df(df, param_cols, param_types):
    for col, col_type in zip(param_cols, param_types):
        df[col] = df[col].astype(col_type)
    return df


def _convert_to_param_space(df, param_cols, param_types):
    dimensions = []
    for colname, col_type in zip(param_cols, param_types):
        if col_type == str:
            dimensions.append(skopt.space.Categorical(categories=df[colname].unique(),
                                                      name=colname))
        elif col_type == float:
            low, high = df[colname].min(), df[colname].max()
            dimensions.append(skopt.space.Real(low, high, name=colname))
        else:
            raise NotImplementedError
    skopt_space = skopt.Space(dimensions)
    return skopt_space


def _convert_space_hop_skopt(space):
    dimensions = []
    for name, specs in space.items():
        specs = str(specs).split('\n')
        method = specs[3].split(' ')[-1]
        bounds = specs[4:]
        if len(bounds) == 1:
            bounds = bounds[0].split('range')[-1]
            bounds = bounds.replace('(', '').replace(')', '').replace('}', '')
            low, high = [float(v) for v in bounds.split(',')]
        else:
            vals = [float(b.split('Literal')[-1].replace('}', '').replace('{', ''))
                    for b in bounds]
            low = min(vals)
            high = max(vals)
        if method == 'randint':
            dimensions.append(skopt.space.Integer(low, high, name=name))
        elif method == 'uniform':
            dimensions.append(skopt.space.Real(low, high, name=name, prior='uniform'))
        elif method == 'loguniform':
            low, high = np.exp(low), np.exp(high)
            dimensions.append(skopt.space.Real(low, high, name=name, prior='log-uniform'))
        else:
            raise NotImplementedError
    skopt_space = skopt.Space(dimensions)
    return skopt_space
