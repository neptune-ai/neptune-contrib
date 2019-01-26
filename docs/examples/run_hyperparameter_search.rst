
Run skopt/hyperopt hyper parameter sweep in neptune
===================================================

Prerequisites
-------------

Training and evaluation script
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We will assume that you have python script that runs model training and
evaluation based on the parameters defined via neptune context.

.. code:: ipython3

    import neptune
    import lightgbm as lgb
    from sklearn.datasets import load_wine
    from sklearn.model_selection import train_test_split
    
    ctx = neptune.Context()
    
    data = load_wine()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target,
                                                        test_size=0.2, random_state=1234)
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
    
    params = {'boosting_type': ctx.params.boosting_type,
              'objective': ctx.params.objective,
              'num_class': ctx.params.num_class,
              'num_leaves': ctx.params.num_leaves,
              'max_depth': ctx.params.max_depth,
              'learning_rate': ctx.params.learning_rate,
              'feature_fraction': ctx.params.feature_fraction}
    
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=ctx.params.num_boost_round,
                    valid_sets=[lgb_train, lgb_eval],
                    valid_names=['train', 'valid'],
                    )
    
    eval_loss= gbm.best_score['valid']['multi_logloss']
    ctx.properties['eval_loss'] = eval_loss

Note that the evaluation metric should be saved as a neptune property:

.. code:: python

    ctx.properties['YOUR_EVALUATION_METRIC']=score

Configuration file
~~~~~~~~~~~~~~~~~~

This is not strictly necessary but it makes things cleaner. It is a good
idea to define hyperparameters and properties in a neptune configuration
file.

Let's call it ``neptune.yaml``

.. code:: ipython3

    project: neptune-ml/neptune-examples
    
    metric:
      channel: 'eval_loss'
      goal: minimize
    
    parameters:
        boosting_type: 'gbdt'
        objective: 'multiclass'
        num_class: 3
        num_boost_round: 10
        learning_rate: 0.1
        num_leaves: 10
        max_depth: 10
        feature_fraction: 0.9

Scikit-Optimize parameter sweep
-------------------------------

Imports
~~~~~~~

import neptune import skopt import neptunecontrib.hpo.utils as hp\_utils
import neptunecontrib.monitoring.skopt as sk\_monitor

ctx = neptune.Context() ctx.tags.append('skopt\_forest\_search')

METRIC\_CHANNEL\_NAME = 'eval\_loss' PROJECT\_NAME =
'neptune-ml/neptune-examples' N\_CALLS = 50 N\_RANDOM\_STARTS = 10

Define search space
~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    space = [skopt.space.Integer(10, 60, name='num_leaves'),
             skopt.space.Integer(2, 30, name='max_depth'),
             skopt.space.Real(0.1, 0.9, name='feature_fraction', prior='uniform')]

Define objective function
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    @skopt.utils.use_named_args(space)
    def objective(**params):
        return hp_utils.make_objective(params,
                                       command=['neptune run --config neptune.yaml','train_evaluate.py'],
                                       metric_channel_name=METRIC_CHANNEL_NAME,
                                       project_name=PROJECT_NAME)

.. code:: ipython3

    ### Define NeptuneMonitor to observe metrics during training

.. code:: ipython3

    monitor = sk_monitor.NeptuneMonitor(ctx)

Run skopt optimization
~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    results = skopt.forest_minimize(objective, space, callback=[monitor],
                                    base_estimator='ET',
                                    n_calls=N_CALLS,
                                    n_random_starts=N_RANDOM_STARTS)

Log best parameters and diagnostic charts to Neptune
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    ctx.channel_send(METRIC_CHANNEL_NAME, results.fun)
    sk_monitor.send_best_parameters(results, ctx)
    sk_monitor.send_plot_convergence(results, ctx)
    sk_monitor.send_plot_evaluations(results, ctx)
    sk_monitor.send_plot_objective(results, ctx)

Hyperopt parameter sweap
------------------------

Imports
~~~~~~~

.. code:: ipython3

    from collections import OrderedDict
    
    from hyperopt import hp, tpe, fmin, Trials
    import neptune
    import skopt
    from sklearn.externals import joblib
    import neptunecontrib.hpo.utils as hp_utils
    import neptunecontrib.monitoring.skopt as sk_monitor
    
    ctx = neptune.Context()
    ctx.tags.append('tpe_search')
    
    METRIC_CHANNEL_NAME = 'eval_loss'
    PROJECT_NAME = 'neptune-ml/neptune-examples'
    TRIALS_PATH = 'trials.pkl'
    N_CALLS = 50

Define search space
~~~~~~~~~~~~~~~~~~~

Normally you define your search space in hyperopt by simply creating a
dict. However, we want to make sure that the names are in the same order
to be able to do some formatting later.

.. code:: ipython3

    space = OrderedDict(num_leaves=hp.choice('num_leaves', range(10, 60, 1)),
                        max_depth=hp.choice('max_depth', range(2, 30, 1)),
                        feature_fraction=hp.uniform('feature_fraction', 0.1, 0.9)
                       )

Define objective function
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    def objective(params):
        return hp_utils.make_objective(params,
                                       command=['neptune run --config neptune.yaml','train_evaluate.py'],
                                       metric_channel_name=METRIC_CHANNEL_NAME,
                                       project_name=PROJECT_NAME)

Run hyperopt optimization
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    trials = Trials()
    _ = fmin(objective, space, trials=trials, algo=tpe.suggest, max_evals=N_CALLS)

Log best parameters and diagnostic charts to Neptune
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Convert ``hyperopt.Trials`` object into
``scipy.optimize.OptimizeResult``

.. code:: ipython3

    results = hp_utils.hyperopt2skopt(trials, space)

Send parameters and diagnostic charts to neptune

.. code:: ipython3

    ctx.channel_send(METRIC_CHANNEL_NAME, results.fun)
    sk_monitor.send_runs(results, ctx)
    sk_monitor.send_best_parameters(results, ctx)
    sk_monitor.send_plot_convergence(results, ctx)
    sk_monitor.send_plot_evaluations(results, ctx)
