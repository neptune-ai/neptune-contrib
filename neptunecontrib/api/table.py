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

import neptune

__all__ = [
    'log_table',
]

def log_table(name, table, experiment=None):
    """Logs pandas dataframe to neptune.

    Pandas dataframe is converted to an HTML table and logged to Neptune as an artifact with path tables/{name}.html

    Args:
        name (:obj:`str`):
            | Name of the chart (without extension) that will be used as a part of artifact's destination.
        table (:obj:`pandas.Dataframe`):
            | DataFrame table
        experiment (:obj:`neptune.experiments.Experiment`, optional, default is ``None``):
            | For advanced users only. Pass Neptune
              `Experiment <https://docs.neptune.ai/neptune-client/docs/experiment.html#neptune.experiments.Experiment>`_
              object if you want to control to which experiment data is logged.
            | If ``None``, log to currently active, and most recent experiment.

    Examples:
        Start an experiment::

            import neptune

            neptune.init(api_token='ANONYMOUS',
                         project_qualified_name='shared/showroom')
            neptune.create_experiment(name='experiment_with_tables')

        Create or load dataframe::

            import pandas as pd

            iris_df = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv', nrows=100)

        Log it to Neptune::

             from neptunecontrib.api import log_table

             log_table('pandas_df', iris_df)

        Check out how the logged table looks in Neptune:
        https://ui.neptune.ai/o/shared/org/showroom/e/SHOW-977/artifacts?path=tables%2F&file=pandas_df.html
     """
    _exp = experiment if experiment else neptune

    _exp.log_artifact(export_pandas_dataframe(table), "tables/" + name + '.html')


def export_pandas_dataframe(table):
    from io import StringIO

    buffer = StringIO(table.to_html())
    buffer.seek(0)

    return buffer
