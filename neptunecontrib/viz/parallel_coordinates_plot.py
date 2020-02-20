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


def make_parallel_coordinates_plot(columns=None,
                                   html_file_path=None,
                                   id=None,
                                   state=None,
                                   owner=None,
                                   tag=None,
                                   min_running_time=None):
    """Visualize experiments on the parallel coordinates plot.

    Make interactive parallel coordinates plot to analyse multiple experiments.
    This function, when executed in Notebook cell,
    displays interactive parallel coordinates plot with all selected experiments.
    Another option is to save visualization to the standalone html file - see examples below.

    You can also inspect the lineage of experiments.
    `See example <https://neptune-contrib.readthedocs.io/examples/hiplot_visualizations.html>`_

    This visualization is build using HiPlot - library published by the Facebook AI group.
    Link to HiPlot docs: https://facebookresearch.github.io/hiplot/index.html

    Learn more about parallel coordinates plot here: https://en.wikipedia.org/wiki/Parallel_coordinates

    Args:
        columns (:obj:`list` of :obj:`str`, optional, default is ``None``):
            | Columns to display on the plot.
            | If `None`, then experiment id like `SAN-12`, experiment `owner` and all parameters are used as columns.
        html_file_path (:obj:`str`, optional, default is ``None``):
            | Saves visualization as a standalone html file. No external dependencies needed.
        id (:obj:`str` or :obj:`list` of :obj:`str`, optional, default is ``None``):
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

            # make visualization for all experiments in project
            make_parallel_coordinates_plot()

            # make visualization for experiment with tag 'segmentation' and save to html file.
            make_parallel_coordinates_plot(html_file_path='visualizations.html', tag='segmentation')

            # make visualization for all experiments created by john and use selected columns
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

    df = neptune.project.get_leaderboard(id=id, state=state, owner=owner, tag=tag, min_running_time=min_running_time)
    assert df.shape[0] != 0, 'No experiments to show. Try other filters.'

    if columns is None:
        columns = ['id', 'owner']
        for col_name in df.columns.to_list():
            if 'parameter_' in col_name:
                columns.append(col_name)
    elif isinstance(columns, str):
        assert columns in df.columns.to_list(), f'There is no "{columns}" in the project columns.'
        columns = ['id', columns]
    elif isinstance(columns, list):
        if 'id' not in columns:
            columns.append('id')
        assert all(column in df.columns.to_list() for column in columns), \
            'Check "columns" parameter, for columns that are not in this project.'
    else:
        raise TypeError(f'{columns} must be None, string or list of string')

    # sort experiments by neptune id
    df = df[columns]
    df = df.rename(columns={'id': 'neptune_id'})
    _exp_ids_series = df['neptune_id'].apply(lambda x: int(x.split('-')[-1]))
    df.insert(loc=0, column='exp_number', value=_exp_ids_series)
    df = df.sort_values(by='exp_number', ascending=True)
    df = df.astype({'neptune_id': 'string'})

    # prepare HiPlot visualization
    input_to_hiplot = df.T.to_dict().values()
    hiplot_vis = hip.Experiment().from_iterable(input_to_hiplot)
    for j, datapoint in enumerate(hiplot_vis.datapoints[1:], 1):
        datapoint.from_uid = hiplot_vis.datapoints[j-1].uid

    # save to html if requested
    if html_file_path is not None:
        assert isinstance(html_file_path, str),\
            f'"html_file_path" should be string, but {type(html_file_path)} is given'
        if len(os.path.dirname(html_file_path)) > 0:
            os.makedirs(os.path.dirname(html_file_path), exist_ok=True)
        hiplot_vis.to_html(html_file_path)
    hiplot_vis.display()
