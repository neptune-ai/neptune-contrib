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


def neptune_monitor(experiment=None, prefix=''):
    """Logs lightGBM learning curves to Neptune.

    Goes over the list of metrics and valid_sets passed to the `lgb.train`
    object and logs them to a separate channels. For example with 'objective': 'multiclass'
    and `valid_names=['train','valid']` there will be 2 channels created:
    `train_multiclass_logloss` and `valid_multiclass_logloss`.

    Args:
        ctx(`neptune.Context`): Neptune context.
        prefix(str): Prefix that should be added before the `metric_name`
            and `valid_name` before logging to the appropriate channel.

    Returns:
       `func`: Callback function that should be passed to the `callbacks` parameter of
          the `lgb.train` function.

    Examples:
        Prepare dataset:

        >>> import lightgbm as lgb
        >>> from sklearn.model_selection import train_test_split
        >>> from sklearn.datasets import load_wine
        >>> data = load_wine()
        >>> X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.1)
        >>> lgb_train = lgb.Dataset(X_train, y_train)
        >>> lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

        Define model parameters:

        >>> params = {'boosting_type': 'gbdt',
                      'objective': 'multiclass',
                      'num_class': 3,
                      'num_leaves': 31,
                      'learning_rate': 0.05,
                      'feature_fraction': 0.9
                      }

        Define your Neptune monitor:

        >>> monitor = neptune_monitor()

        Run `lgb.train` passing `neptune_monitor()` to the `callbacks` parameter:

        >>> gbm = lgb.train(params,
        >>>                 lgb_train,
        >>>                 num_boost_round=500,
        >>>                 valid_sets=[lgb_train, lgb_eval],
        >>>                 valid_names=['train','valid'],
        >>>                 callbacks=[monitor],
        >>>                )

    Note:
        If you are running a k-fold validation it is a good idea to add the k-fold prefix
        and pass it to the `neptune_monitor` function:

        >>> prefix='fold{}_'.format(fold_id)
        >>> monitor = neptune_monitor(prefix)
    """

    _exp = experiment if experiment else neptune

    def callback(env):
        for name, loss_name, loss_value, _ in env.evaluation_result_list:
            channel_name = '{}{}_{}'.format(prefix, name, loss_name)
            _exp.send_metric(channel_name, x=env.iteration, y=loss_value)

    return callback
