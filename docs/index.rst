neptune-contrib: open-source extensions build on top of neptune
==========================
   
neptune-contrib is an open-source library build on top of `neptune.ml` and
`neptune-lib`, that gives you a toolkit of helpers and extensions that 
make working with neptune.ml easier. The code is available on `GitHub <http://github.com/neptune-ml/neptune-contrib>`_. 

With neptune-contrib you can interactively compare experiments, vizualize 
hyperparameters, log matplotlib charts to neptune, run bayesian hyper parameter sweeps and more.

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   getting_started/overview
   getting_started/installation
   getting_started/starting
   
.. toctree::
   :maxdepth: 2   
   :caption: api:

   Utils <api_utils>

.. toctree::
   :maxdepth: 2   
   :caption: hyper parameter optimization:

   Utils <hpo_utils>
   
.. toctree::
   :maxdepth: 2   
   :caption: monitoring:
   
   LightGBM <monitoring_lightgbm>
   Local Notebooks <monitoring_notebooks>
   Scikit-Optimize <monitoring_skopt>
   Utils <monitoring_utils>
   
.. toctree::
   :maxdepth: 2   
   :caption: vizualizations:

   Experiments <viz_experiments>
   Utils <viz_utils>
   
Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
