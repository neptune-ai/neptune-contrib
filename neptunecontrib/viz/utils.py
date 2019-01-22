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

from scipy.optimize import OptimizeResult
import skopt


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
