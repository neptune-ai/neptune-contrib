
Interactive Experiments Compare
===============================

Concatenate data from multiple experiment runs
----------------------------------------------

Use the ``concat_experiments_on_channel`` helper function from
neptune-contrib to do that.

.. code:: ipython3

    from neptunecontrib.api.utils import concat_experiments_on_channel
    
    exp_df = concat_experiments_on_channel(experiments, channel_name)
    exp_df.head()




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
          <th>x</th>
          <th>unet_0 epoch_val iout loss</th>
          <th>id</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>0.0</td>
          <td>0.612894</td>
          <td>SAL-1134</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.0</td>
          <td>0.679910</td>
          <td>SAL-1134</td>
        </tr>
        <tr>
          <th>2</th>
          <td>2.0</td>
          <td>0.690255</td>
          <td>SAL-1134</td>
        </tr>
        <tr>
          <th>3</th>
          <td>3.0</td>
          <td>0.707496</td>
          <td>SAL-1134</td>
        </tr>
        <tr>
          <th>4</th>
          <td>4.0</td>
          <td>0.713793</td>
          <td>SAL-1134</td>
        </tr>
      </tbody>
    </table>
    </div>



Plot an interactive Altair chart.
---------------------------------

Use ``channel_curve_compare`` from ``neptunecontrib.viz``.

.. code:: ipython3

    from neptunecontrib.viz.experiments import channel_curve_compare
    
    channel_curve_compare(exp_df)




.. image:: interactive_compare_experiments_files/interactive_compare_experiments_3_0.png



**Note** You may need to change the rendering method depending on your
machine.

.. code:: python

    import altair as alt
    alt.renderers.enable('jupyterlab')

Because Vega-Lite visualizations keep all the chart data in the HTML the
visualizations can consume huge amounts of memory if not handled
properly. That is why, by default the hard limit of 5000 rows is set to
the len of dataframe. That being said, you can disable it by adding the
following line in the notebook or code.

.. code:: python

    import altair as alt
    alt.data_transformers.enable('default', max_rows=None)
