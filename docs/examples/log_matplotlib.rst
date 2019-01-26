
Log matplotlib figure to neptune
================================

Create figure
-------------

.. code:: ipython3

    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    fig = plt.figure(figsize=(16,12))
    sns.distplot(np.random.random(100))

Convert to PIL
--------------

.. code:: ipython3

    from neptunecontrib.monitoring.uitls import fig2pil
    
    pil_figure = fig2pil(fig)

Creatae ``neptune.Image``
-------------------------

.. code:: ipython3

    import neptune
    
    neptune_figure = neptune.Image(name='chart', description='', data=pil_figure)

Log to neptune
--------------

.. code:: ipython3

    ctx = neptune.Context()
    
    ctx.channel_send('chart_logs', neptune_figure)

Explore in neptune
------------------

.. figure:: 
   :alt: image

   image
