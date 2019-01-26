
Scikit-Optimize plot\_evaluations
=================================

Convert hyper parameter dataframe to a OptimizeResult format
------------------------------------------------------------

Say you have your hyper parameter and metric stored in a dataframe.

.. code:: ipython3

    hyper_df.head()




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>ROC_AUC</th>
          <th>lgbm__max_depth</th>
          <th>lgbm__num_leaves</th>
          <th>lgbm__min_child_samples</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>0.7578376974936794</td>
          <td>20.0</td>
          <td>50.0</td>
          <td>20.0</td>
        </tr>
        <tr>
          <th>1</th>
          <td>0.7578376974936794</td>
          <td>20.0</td>
          <td>50.0</td>
          <td>20.0</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.7578376974936794</td>
          <td>20.0</td>
          <td>50.0</td>
          <td>20.0</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.7383150842338956</td>
          <td>20.0</td>
          <td>50.0</td>
          <td>20.0</td>
        </tr>
        <tr>
          <th>4</th>
          <td>0.7859497222152486</td>
          <td>-1.0</td>
          <td>100.0</td>
          <td>600.0</td>
        </tr>
      </tbody>
    </table>
    </div>



You can use ``df2result`` helper function from ``neptunecontrib.viz``.

.. code:: ipython3

    from neptunecontrib.viz.utils import df2result
    
    result = df2result(hyper_df, 
                       metric_col='ROC_AUC', 
                       param_cols=['lgbm__max_depth',
                                   'lgbm__num_leaves',
                                   'lgbm__min_child_samples'])
    type(result), result.keys()




.. parsed-literal::

    (scipy.optimize.optimize.OptimizeResult,
     dict_keys(['x_iters', 'func_vals', 'x', 'fun', 'space']))



Use skopt.plots
---------------

Now you can use functions from ``skopt.plots`` with no problems.

.. code:: ipython3

    import matplotlib.pyplot as plt
    
    from skopt.plots import plot_evaluations
    
    eval_plot = plot_evaluations(result, bins=20)
    eval_plot;



.. image:: explore_hyperparams_skopt_files/explore_hyperparams_skopt_5_0.png


**Note**

This chart is actually in a pretty weird format. It's an array of
``matplotlib.axes`` objects.

You can convert it to the standard matplotlib Figure by using a helper
function from ``neptunecontrib.viz``.

.. code:: ipython3

    from neptunecontrib.viz.utils import axes2fig
    
    fig = axes2fig(eval_plot)
    type(fig);



.. image:: explore_hyperparams_skopt_files/explore_hyperparams_skopt_7_0.png

