
Log lightGBM metrics to neptune
===============================

Prerequisites
-------------

Create your dataset and define parameters

.. code:: ipython3

    import lightgbm as lgb
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import load_wine
    data = load_wine()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.1)
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
    
    params = {'boosting_type': 'gbdt',
                  'objective': 'multiclass',
                  'num_class': 3,
                  'num_leaves': 31,
                  'learning_rate': 0.05,
                  'feature_fraction': 0.9
                  }

Create ``neptune_monitor`` callback
-----------------------------------

.. code:: ipython3

    from neptunecontrib.monitoring.lightgbm import neptune_monitor
    
    monitor = neptune_monitor()

Add ``neptune_monitor`` callback to ``lgb.train``
-------------------------------------------------

.. code:: ipython3

    gbm = lgb.train(params,
            lgb_train,
            num_boost_round=500,
            valid_sets=[lgb_train, lgb_eval],
            valid_names=['train','valid'],
            callbacks=[monitor],
           )

Monitor your lightGBM training in neptune
-----------------------------------------
