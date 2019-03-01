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

import random
import string
import subprocess

from neptunelib.session import Session
import pandas as pd
from scipy.optimize import OptimizeResult
import skopt
from retrying import retry


def make_objective(param_set, command, metric_channel_name, project_name, tag='trash'):
    """Converts `neptune run` script to a python function that returns a score.

    Helper function that wraps your train_evaluate_model.py script into a function
    that takes hyper parameters as input and returns the score on a specified metric.

    Args:
        param_set(dict): Dictionary of parameter name parameter value. The objective
            function is evaluated on those parameters.
        command(list):bash neptune command represented as a list. It is assumed
            that the .py script path is a last element.
            E.g. ['neptune run --config neptune.yaml', 'main.py']
        metric_channel_name(str): name of the property where the evaluation
            result is stored. It is crucial that the single run result is stored
            as property.E.g. ctx.properties[metric_channel_name] = 0.8.
        project_name(str): NAME_SPACE/PROJECT_NAME. E.g. 'neptune-ml/neptune-examples'
        tag(str): Tag that should be added to the single run experiment.
            It is useful to add this tag so that you can group or get rid of the single run
            experiments in your project. Default is 'trash'.

    Returns:
        float: Score that was reported in the `ctx.properties[metric_channel_name]`
        of the run of train_evaluate_model.py script.


    Examples:
        Prepare the space of hyperparameters to search over.

        >>> space = [skopt.space.Integer(10, 60, name='num_leaves'),
                     skopt.space.Integer(2, 30, name='max_depth'),
                     skopt.space.Real(0.1, 0.9, name='feature_fraction', prior='uniform')]

        Define the objective.

        >>> @skopt.utils.use_named_args(space)
            def objective(**params):
                return hp_utils.make_objective(params,
                                               command=['neptune run --config neptune.yaml',
                                               'train_evaluate.py'],
                                               metric_channel_name='valid_accuracy',
                                               project_name='neptune-ml/neptune-examples')

        Run skopt training.

        >>> results = skopt.forest_minimize(objective, space,
                                base_estimator='ET', n_calls=100, n_random_starts=10)

    """
    command, exp_id_tag = _create_command_and_id_tag(command, param_set, tag)

    subprocess.call(command, shell=True)

    score = _get_score(exp_id_tag, metric_channel_name, project_name)
    return score


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
    optimize_results.x = list(trials.argmin.values())
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

        >>> from neptunelib.api.session import Session
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
    results_['target'] = -1.0 * results['value']
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
            dimensions.append(skopt.space.Real(low, high, name=name, prior='log-uniform'))
        else:
            raise NotImplementedError
    skopt_space = skopt.Space(dimensions)
    return skopt_space


def _create_command_and_id_tag(command, param_set, tag):
    command.insert(-1, '--tag {}'.format(tag))
    for name, value in param_set.items():
        command.insert(-1, "--parameter {}:{}".format(name, value))

    exp_id_tag = _get_random_string()
    command.insert(-1, '--tag {}'.format(exp_id_tag))

    command = " ".join(command)
    return command, exp_id_tag


def _get_random_string(length=64):
    x = ''.join(random.choice(string.ascii_lowercase + string.digits)
                for _ in range(length))
    return x


@retry(stop_max_attempt_number=50, wait_random_min=1000, wait_random_max=5000)
def _get_score(exp_id_tag, metric_name, project_name):
    namespace = project_name.split('/')[0]

    session = Session()
    project = session.get_projects(namespace)[project_name]
    experiment = project.get_experiments(tag=[exp_id_tag])[0]
    score = float(experiment.properties[metric_name].tolist()[0])
    return score
