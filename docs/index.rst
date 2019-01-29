neptune-contrib: open-source contributions to Neptune.ml
===========================================

This library is a collection of helpers and extensions that make working
with `neptune.ml` more effective and better. It is build on top of neptune-cli
and neptune-lib and gives you option to do things like:
 * interactive visualizations of experiment runs or hyperparameters
 * running hyper parameter sweeps in scikit-optimize, hyperopt or any other tool you like
 * monitor training of the lightGBM models
 * much more
 
And the best thing is you can extend it yourself or... tell us to do it for you :).

.. toctree::
   :maxdepth: 1
   :caption: Get Started

   installation
   overview

.. toctree::
   :maxdepth: 1
   :caption: Examples

   examples/examples_index

.. toctree::
   :maxdepth: 1
   :caption: User Guide

   api.utils <user_guide/api/utils>
   hpo.utils <user_guide/hpo/utils>
   monitoring.notebooks <user_guide/monitoring/notebooks>
   monitoring.lightgbm <user_guide/monitoring/lightgbm>
   monitoring.skopt <user_guide/monitoring/skopt>
   monitoring.utils <user_guide/monitoring/utils>
   viz.experiments <user_guide/viz/experiments>
   viz.utils <user_guide/viz/utils>
   

Bug Reports and Questions
-----------------------

neptune-contrib is MIT-licensed and the source code is available on `GitHub`_. If you
find yourself in any trouble drop an isse on `Git Issues`_, fire a feature request on
`Git Feature Request`_ or ask us on the `Neptune community forum`_ or `Neptune Slack`_.


Contribute
-----------------------

We keep an updated list of open issues/feature ideas on github project page `Github project`_.
If you feel like taking a shot at one of those do go for it!
In case of any trouble please talk to us on the `Neptune Slack`_.


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`


.. _GitHub: http://github.com/neptune-ml/neptune-contrib
.. _Git Issues: http://github.com/neptune-ml/neptune-contrib/issues
.. _Git Feature Request: http://github.com/neptune-ml/neptune-contrib/issues
.. _Neptune community forum: http://community.neptune.ml
.. _Github project: http://github.com/neptune-ml/neptune-contrib/projects
.. _Neptune Slack: http://slack/neptune-community.com
