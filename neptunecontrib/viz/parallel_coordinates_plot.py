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

import hiplot as hip
import neptune


def make_parallel_coordinates_plot(html_file_path=None,
                                   metrics=None,
                                   params=None,
                                   properties=None,
                                   experiment_id=None,
                                   state=None,
                                   owner=None,
                                   tag=None,
                                   min_running_time=None):
    """Visualize experiments on the parallel coordinates plot.

    Make interactive parallel coordinates plot to analyse multiple experiments.
    This function, when executed in Notebook cell,
    displays interactive parallel coordinates plot in the cell's output.
    Another option is to save visualization to the standalone html file.
    You can also inspect the lineage of experiments.

    **See** `example <https://neptune-contrib.readthedocs.io/examples/hiplot_visualizations.html>`_
    **for full use case.**

    Use ``metrics``, ``params`` and ``properties`` arguments to select what data you want to see as axes.

    Use ``experiment_id``, ``state``, ``owner``, ``tag``, ``min_running_time`` arguments to filter experiments
    included in a plot. Only experiments matching all the criteria will be returned.

    This visualization it built using `HiPlot <https://facebookresearch.github.io/hiplot/index.html>`_.
    It is a library published by the Facebook AI group.
    Learn more about the `parallel coordinates plot <https://en.wikipedia.org/wiki/Parallel_coordinates>`_.

    Note:
        Make sure you have your project set: `neptune.init('USERNAME/example-project')`

    Args:
        html_file_path (:obj:`str`, optional, default is ``None``):
            | Saves visualization as a standalone html file. No external dependencies needed.
        metrics (:obj:`str` or :obj:`list` of :obj:`str`, optional, default is ``None``):
            | Metrics to display on the plot (as axes).
            | If `None`, then display all metrics.
            | If empty list `[]`, then exclude all metrics.
        params (:obj:`str` or :obj:`list` of :obj:`str`, optional, default is ``None``):
            | Parameters to display on the plot (as axes).
            | If `None`, then display all parameters.
            | If empty list `[]`, then exclude all parameters.
        properties (:obj:`str` or :obj:`list` of :obj:`str`, optional, default is ``None``):
            | Properties to display on the plot (as axes).
            | If `None`, then display all properties.
            | If empty list `[]`, then exclude all properties.
        experiment_id (:obj:`str` or :obj:`list` of :obj:`str`, optional, default is ``None``):
            | An experiment id like ``'SAN-1'`` or list of ids like ``['SAN-1', 'SAN-2']``.
            | Matching any element of the list is sufficient to pass criterion.
        state (:obj:`str` or :obj:`list` of :obj:`str`, optional, default is ``None``):
            | An experiment state like ``'succeeded'`` or list of states like ``['succeeded', 'running']``.
            | Possible values: ``'running'``, ``'succeeded'``, ``'failed'``, ``'aborted'``.
            | Matching any element of the list is sufficient to pass criterion.
        owner (:obj:`str` or :obj:`list` of :obj:`str`, optional, default is ``None``):
            | *Username* of the experiment owner (User who created experiment is an owner) like ``'josh'``
              or list of owners like ``['frederic', 'josh']``.
            | Matching any element of the list is sufficient to pass criterion.
        tag (:obj:`str` or :obj:`list` of :obj:`str`, optional, default is ``None``):
            | An experiment tag like ``'lightGBM'`` or list of tags like ``['pytorch', 'cycleLR']``.
            | Only experiments that have all specified tags will match this criterion.
        min_running_time (:obj:`int`, optional, default is ``None``):
            Minimum running time of an experiment in seconds, like ``2000``.

    Examples:

        .. code:: python3

            # Make sure you have your project set:
            neptune.init('USERNAME/example-project')

            # (example 1) make visualization for all experiments in project
            make_parallel_coordinates_plot()

            # (example 2) make visualization for experiment with tag 'segmentation' and save to html file.
            make_parallel_coordinates_plot(html_file_path='visualizations.html', tag='segmentation')

            # (example 3) make visualization for all experiments created by john and use selected columns
            make_parallel_coordinates_plot(
                columns=['channel_epoch_acc', 'channel_epoch_loss',
                         'parameter_lr', 'parameter_dense', 'parameter_dropout'],
                owner='john')
    """
    if neptune.project is None:
        msg = """You do not have project, from which to fetch data.
                 Use neptune.init() to set project, for example: neptune.init('USERNAME/example-project').
                 See docs: https://docs.neptune.ai/neptune-client/docs/neptune.html#neptune.init"""
        raise ValueError(msg)

    df = neptune.project.get_leaderboard(id=experiment_id,
                                         state=state,
                                         owner=owner,
                                         tag=tag,
                                         min_running_time=min_running_time)
    assert df.shape[0] != 0, 'No experiments to show. Try other filters.'

    if columns is None:
        columns = ['id', 'owner']
        for col_name in df.columns.to_list():
            if 'parameter_' in col_name:
                columns.append(col_name)
    elif isinstance(columns, str):
        assert columns in df.columns.to_list(), 'There is no "{}" in the project columns.'.format(columns)
        columns = ['id', columns]
    elif isinstance(columns, list):
        if 'id' not in columns:
            columns.append('id')
        assert all(column in df.columns.to_list() for column in columns), \
            '"columns" parameter contains columns that are not in selected experiments.'
    else:
        raise TypeError('{} must be None, string or list of string'.format(columns))

    # Sort experiments by neptune id
    df = df[columns]
    df = df.rename(columns={'id': 'neptune_id'})
    _exp_ids_series = df['neptune_id'].apply(lambda x: int(x.split('-')[-1]))
    df.insert(loc=0, column='exp_number', value=_exp_ids_series)
    df = df.sort_values(by='exp_number', ascending=True)

    # Prepare HiPlot visualization
    input_to_hiplot = df.T.to_dict().values()
    hiplot_vis = hip.Experiment().from_iterable(input_to_hiplot)
    for j, datapoint in enumerate(hiplot_vis.datapoints[1:], 1):
        datapoint.from_uid = hiplot_vis.datapoints[j-1].uid

    # Save to html if requested
    if html_file_path is not None:
        assert isinstance(html_file_path, str),\
            '"html_file_path" should be string, but {} is given'.format(type(html_file_path))
        if os.path.dirname(html_file_path):
            os.makedirs(os.path.dirname(html_file_path), exist_ok=True)
        hiplot_vis.to_html(html_file_path)
    hiplot_vis.display()
