#
# Copyright (c) 2020, Neptune Labs Sp. z o.o.
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
import os
import tempfile

import neptune
import xgboost as xgb


def neptune_monitor(log_model=True,
                    log_importance=True,
                    log_tree=None,
                    **kwargs):
    """XGBoost-monitor for Neptune experiments.

    Args:
        log_model (:obj:`bool`, optional, default is ``False``):
            | Log booster to Neptune at the end of training.
        log_importance (:obj:`bool`, optional, default is ``False``):
            | Log feature importance to Neptune as image at the end of training.
        log_tree (:obj:`int` or :obj:`list` of :obj:`int`, optional, default is ``None``):
            | Log specified trees to Neptune as images at the end of training.


    Returns:

    """
    assert isinstance(log_model, bool),\
        'log_model must be bool, got {} instead. Check log_model parameter.'.format(type(log_model))
    assert isinstance(log_importance, bool),\
        'log_importance must be bool, got {} instead. Check log_importance parameter.'.format(type(log_importance))
    if log_tree is not None:
        assert isinstance(log_tree, int) or isinstance(log_tree, list),\
            'log_tree must be int or list of int, got {} instead. Check log_tree parameter.'.format(type(log_tree))

    def callback(env):
        # Log metrics after iteration
        for k, v in env.evaluation_result_list:
            neptune.log_metric(k, v)

        # End of training
        # Log booster
        if env.iteration == env.end_iteration and log_model:
            with tempfile.TemporaryDirectory(dir='.') as d:
                path = os.path.join(d, 'bst.model')
                env.model.save_model(path)
                neptune.log_artifact(path)

        # Log feature importance
        if env.iteration + 1 == env.end_iteration and log_importance:
            importance = xgb.plot_importance(env.model, max_num_features=kwargs['max_num_features'])
            neptune.log_image('feature_importance', importance.figure)

        # Log trees
        if env.iteration + 1 == env.end_iteration and log_tree is not None:
            with tempfile.TemporaryDirectory(dir='.') as d:
                for i in list(log_tree):
                    file_name = 'tree_{}'.format(i)
                    tree = xgb.to_graphviz(booster=env.model, num_trees=i)
                    tree.render(filename=file_name, directory=d, view=False, format='png')
                    neptune.log_image('trees',
                                      os.path.join(d, '{}.png'.format(file_name)),
                                      image_name=file_name)
    return callback


# ToDo tree figure size
# ToDo docstrings
# ToDo larges example data
# ToDo document kwargs
# ToDO make sure graphviz or neptune errors will not crash exp
