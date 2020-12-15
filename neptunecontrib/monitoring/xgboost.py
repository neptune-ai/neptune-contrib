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


def neptune_callback(log_model=True,
                     log_importance=True,
                     max_num_features=None,
                     log_tree=None,
                     experiment=None,
                     **kwargs):
    """XGBoost callback for Neptune experiments.

    This is XGBoost callback that automatically logs training and evaluation metrics, feature importance chart,
    visualized trees and trained Booster to Neptune.

    Check Neptune documentation for the `full example <https://docs.neptune.ai/integrations/xgboost.html>`_.

    Make sure you created an experiment before you start XGBoost training using ``neptune.create_experiment()``
    (`check our docs <https://docs.neptune.ai/api-reference/neptune/projects/index.html
    #neptune.projects.Project.create_experiment>`_).

    You need to install graphviz and graphviz Python interface for ``log_tree`` feature to work.
    Check `Graphviz <https://graphviz.org/download/>`_ and
    `Graphviz Python interface <https://graphviz.readthedocs.io/en/stable/manual.html#installation>`_
    for installation info.

    Integration works with ``xgboost>=1.2.0``.

    Tip:
        Use this `Google Colab <https://colab.research.google.com//github/neptune-ai/neptune-examples/blob/master/
        integrations/xgboost/docs/Neptune-XGBoost.ipynb>`_
        run it as a "`neptuner`" user - zero setup, it just works.

    Note:
        If you use early stopping, make sure to log model, feature importance and trees on your own.
        Neptune logs these artifacts only after last iteration, which you may not reach because of early stop.

    Args:
        log_model (:obj:`bool`, optional, default is ``True``):
            | Log booster to Neptune after last boosting iteration.
            | If you run xgb.cv, log booster for all folds.
        log_importance (:obj:`bool`, optional, default is ``True``):
            | Log feature importance to Neptune as image after last boosting iteration.
            | Specify number of features using ``max_num_features`` parameter below.
            | If you run xgb.cv, log feature importance for each folds' booster.
        max_num_features (:obj:`int`, optional, default is ``None``):
            | Plot top ``max_num_features`` features on the importance plot.
            | If ``None``, plot all features.
        log_tree (:obj:`list` of :obj:`int`, optional, default is ``None``):
            | Log specified trees to Neptune as images after last boosting iteration.
            | If you run xgb.cv, log specified trees for each folds' booster.
            | Default is ``None`` - do not log any tree.
        experiment (:obj:`neptune.experiments.Experiment`, optional, default is ``None``):
            | For advanced users only. Pass Neptune ``Experiment``
              object if you want to control to which experiment data is logged.
            | If ``None``, log to currently active, and most recent experiment.
        kwargs:
            Parametrize XGBoost functions used in this callback:
            `xgboost.plot_importance <https://xgboost.readthedocs.io/en/latest/python/python_api.html
            ?highlight=plot_tree#xgboost.plot_importance>`_
            and `xgboost.to_graphviz <https://xgboost.readthedocs.io/en/latest/python/python_api.html
            ?highlight=plot_tree#xgboost.to_graphviz>`_.

    Returns:
        :obj:`callback`, function that you can pass directly to the XGBoost callbacks list, for example to the
        ``xgboost.cv()``
        (`see docs <https://xgboost.readthedocs.io/en/latest/python/python_api.html?highlight=plot_tree#xgboost.cv>`_)
        or ``XGBClassifier.fit()``
        (`check docs <https://xgboost.readthedocs.io/en/latest/python/python_api.html?highlight=plot_tree
        #xgboost.XGBClassifier.fit>`_).

    Examples:
        ``xgb.train`` examples

        .. code:: python3

            # basic usage
            xgb.train(param, dtrain, num_round, watchlist,
                      callbacks=[neptune_callback()])

            # do not log model
            xgb.train(param, dtrain, num_round, watchlist,
                      callbacks=[neptune_callback(log_model=False)])

            # log top 5 features' importance chart
            xgb.train(param, dtrain, num_round, watchlist,
                      callbacks=[neptune_callback(max_num_features=5)])

        ``xgb.cv`` examples

        .. code:: python3

            # log 5 trees per each folds' booster
            xgb.cv(param, dtrain, num_boost_round=num_round, nfold=7,
                   callbacks=neptune_callback(log_tree=[0,1,2,3,4]))

            # log only metrics
            xgb.cv(param, dtrain, num_boost_round=num_round, nfold=7,
                   callbacks=[neptune_callback(log_model=False,
                                               log_importance=False,
                                               max_num_features=None,
                                               log_tree=None)])

            # log top 3 features per each folds' booster and first tree
            xgb.cv(param, dtrain, num_boost_round=num_round, nfold=7,
                   callbacks=[neptune_callback(log_model=False,
                                               max_num_features=3,
                                               log_tree=[0,])])

        ``sklearn`` API examples

        .. code:: python3

            # basic usage with early stopping
            xgb.XGBRegressor().fit(X_train, y_train,
                                   early_stopping_rounds=10,
                                   eval_metric=['mae', 'rmse', 'rmsle'],
                                   eval_set=[(X_test, y_test)],
                                   callbacks=[neptune_callback()])

            # do not log model
            clf = xgb.XGBRegressor()
            clf.fit(X_train, y_train,
                    eval_metric=['mae', 'rmse', 'rmsle'],
                    eval_set=[(X_test, y_test)],
                    callbacks=[neptune_callback(log_model=False)])
            y_pred = clf.predict(X_test)

            # log 8 trees
            reg = xgb.XGBRegressor(**params)
            reg.fit(X_train, y_train,
                    eval_metric=['mae', 'rmse', 'rmsle'],
                    eval_set=[(X_test, y_test)],
                    callbacks=[neptune_callback(log_tree=[0,1,2,3,4,5,6,7])])
    """
    if experiment:
        _exp = experiment
    else:
        try:
            neptune.get_experiment()
            _exp = neptune
        except neptune.exceptions.NeptuneNoExperimentContextException:
            raise neptune.exceptions.NeptuneNoExperimentContextException()

    assert isinstance(log_model, bool),\
        'log_model must be bool, got {} instead. Check log_model parameter.'.format(type(log_model))
    assert isinstance(log_importance, bool),\
        'log_importance must be bool, got {} instead. Check log_importance parameter.'.format(type(log_importance))
    if max_num_features is not None:
        assert isinstance(max_num_features, int),\
            'max_num_features must be int, got {} instead. ' \
            'Check max_num_features parameter.'.format(type(max_num_features))
    if log_tree is not None:
        if isinstance(log_tree, tuple):
            log_tree = list(log_tree)
        assert isinstance(log_tree, list),\
            'log_tree must be list of int, got {} instead. Check log_tree parameter.'.format(type(log_tree))

    def callback(env):
        # Log metrics after iteration
        for item in env.evaluation_result_list:
            if len(item) == 2:  # train case
                _exp.log_metric(item[0], item[1])
            if len(item) == 3:  # cv case
                _exp.log_metric('{}-mean'.format(item[0]), item[1])
                _exp.log_metric('{}-std'.format(item[0]), item[2])

        # Log booster, end of training
        if env.iteration + 1 == env.end_iteration and log_model:
            if env.cvfolds:  # cv case
                for i, cvpack in enumerate(env.cvfolds):
                    _log_model(cvpack.bst, 'cv-fold-{}-bst.model'.format(i), _exp)
            else:  # train case
                _log_model(env.model, 'bst.model', _exp)

        # Log feature importance, end of training
        if env.iteration + 1 == env.end_iteration and log_importance:
            if env.cvfolds:  # cv case
                for i, cvpack in enumerate(env.cvfolds):
                    _log_importance(cvpack.bst, max_num_features, _exp, title='cv-fold-{}'.format(i), **kwargs)
            else:  # train case
                _log_importance(env.model, max_num_features, _exp, **kwargs)

        # Log trees, end of training
        if env.iteration + 1 == env.end_iteration and log_tree:
            if env.cvfolds:
                for j, cvpack in enumerate(env.cvfolds):
                    _log_trees(cvpack.bst, log_tree, 'trees-cv-fold-{}'.format(j), _exp, **kwargs)
            else:
                _log_trees(env.model, log_tree, 'trees', _exp, **kwargs)
    return callback


def _log_model(booster, name, npt):
    with tempfile.TemporaryDirectory(dir='.') as d:
        path = os.path.join(d, name)
        booster.save_model(path)
        npt.log_artifact(path)


def _log_importance(booster, max_num_features, npt, **kwargs):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError('Please install matplotlib to log importance')
    importance = xgb.plot_importance(booster, max_num_features=max_num_features, **kwargs) # pylint: disable=E1101
    npt.log_image('feature_importance', importance.figure)
    plt.close('all')


def _log_trees(booster, tree_list, img_name, npt, **kwargs):
    with tempfile.TemporaryDirectory(dir='.') as d:
        for i in tree_list:
            file_name = 'tree_{}'.format(i)
            tree = xgb.to_graphviz(booster=booster, num_trees=i, **kwargs) # pylint: disable=E1101
            tree.render(filename=file_name, directory=d, view=False, format='png')
            npt.log_image(img_name,
                          os.path.join(d, '{}.png'.format(file_name)),
                          image_name=file_name)
