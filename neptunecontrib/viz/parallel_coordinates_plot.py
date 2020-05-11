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
from collections import Counter

import hiplot as hip
import neptune

__all__ = [
    'make_parallel_coordinates_plot',
]

def make_parallel_coordinates_plot(html_file_path=None,
                                   metrics=False,
                                   text_logs=False,
                                   params=True,
                                   properties=False,
                                   experiment_id=None,
                                   state=None,
                                   owner=None,
                                   tag=None,
                                   min_running_time=None):
    """Visualize experiments on the parallel coordinates plot.

    This function, when executed in Notebook, displays interactive parallel coordinates plot in the cell's output.
    Another option is to save visualization to the standalone html file.
    You can also inspect the lineage of experiments.

    **See** `example <https://neptune-contrib.readthedocs.io/examples/hiplot_visualizations.html>`_
    **for the full use case.**

    Axes are ordered as follows: first axis is neptune ``experiment id``,
    second is experiment ``owner``,
    then ``params`` and ``properties`` in alphabetical order.
    Finally, ``metrics`` on the right side (alphabetical order as well).

    This visualization it built using `HiPlot <https://facebookresearch.github.io/hiplot/index.html>`_.
    It is a library published by the Facebook AI group.
    Learn more about the `parallel coordinates plot <https://en.wikipedia.org/wiki/Parallel_coordinates>`_.

    Tip:
        Use ``metrics``, ``params`` and ``properties`` arguments to select what data you want to see as axes.

        Use ``experiment_id``, ``state``, ``owner``, ``tag``, ``min_running_time`` arguments to filter experiments
        included in a plot. Only experiments matching all the criteria will be returned.

    Note:
        Make sure you have your project set: ``neptune.init('USERNAME/example-project')``

    Args:
        html_file_path (:obj:`str`, optional, default is ``None``):
            | Saves visualization as a standalone html file. No external dependencies needed.
        metrics (:obj:`bool` or :obj:`str` or :obj:`list` of :obj:`str`, optional, default is ``False``):
            | Metrics to display on the plot (as axes).
            | If `True`, then display all metrics.
            | If `False`, then exclude all metrics.
        text_logs (:obj:`bool` or :obj:`str` or :obj:`list` of :obj:`str`, optional, default is ``False``):
            | Text logs to display on the plot (as axes).
            | If `True`, then display all text logs.
            | If `False`, then exclude all text logs.
        params (:obj:`bool` or :obj:`str` or :obj:`list` of :obj:`str`, optional, default is ``True``):
            | Parameters to display on the plot (as axes).
            | If `True`, then display all parameters.
            | If `False`, then exclude all parameters.
        properties (:obj:`bool` or :obj:`str` or :obj:`list` of :obj:`str`, optional, default is ``False``):
            | Properties to display on the plot (as axes).
            | If `True`, then display all properties.
            | If `False`, then exclude all properties.
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

    Returns:

        :obj:`ExperimentDisplayed`, object that can be used to get a ``list`` of ``Datapoint`` objects,
        like this: ``ExperimentDisplayed.get_selected()``. This is only implemented for Jupyter notebook. Check
        `HiPlot docs
        <https://facebookresearch.github.io/hiplot/py_reference.html?highlight=display#hiplot.Experiment.display>`_.

    Examples:

        .. code:: python3

            # Make sure you have your project set:
            neptune.init('USERNAME/example-project')

            # (example 1) visualization for all experiments in project
            make_parallel_coordinates_plot()

            # (example 2) visualization for experiment with tag 'optuna' and saving to html file.
            make_parallel_coordinates_plot(html_file_path='visualizations.html', tag='optuna')

            # (example 3) visualization with all params, two metrics for experiment with tag 'optuna'
            make_parallel_coordinates_plot(tag='optuna', metrics=['epoch_accuracy', 'eval_accuracy'])

            # (example 4) visualization with all params and two metrics. All experiments created by john.
            make_parallel_coordinates_plot(metrics=['epoch_accuracy', 'eval_accuracy'], owner='john')
    """
    _all_metrics = []
    _all_text_logs = []
    _all_params = []
    _all_properties = []

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

    # Cast columns to int or str
    for column in df.columns.to_list():
        if column.startswith('channel_'):
            try:
                df = df.astype({column: float})
                _all_metrics.append((column, column.replace('channel_', '')))
            except ValueError:
                df = df.astype({column: str})
                _all_text_logs.append((column, column.replace('channel_', '')))
        elif column.startswith('parameter_'):
            try:
                df = df.astype({column: float})
            except ValueError:
                df = df.astype({column: str})
            _all_params.append((column, column.replace('parameter_', '')))
        elif column.startswith('property_'):
            try:
                df = df.astype({column: float})
            except ValueError:
                df = df.astype({column: str})
            _all_properties.append((column, column.replace('property_', '')))

    # Validate each type of input
    metrics = _validate_input(metrics, _all_metrics, 'metric')
    text_logs = _validate_input(text_logs, _all_text_logs, 'text log')
    params = _validate_input(params, _all_params, 'parameter')
    properties = _validate_input(properties, _all_properties, 'property')

    # Check for name conflicts
    for column in [k for k, v in Counter(metrics + text_logs + params + properties).items() if v > 1]:
        if column in metrics:
            metrics = ['metric__' + column if j == column else j for j in metrics]
            _all_metrics = [(j[0], 'metric__' + column) if j[1] == column else j for j in _all_metrics]
        if column in text_logs:
            text_logs = ['text_log__' + column if j == column else j for j in text_logs]
            _all_text_logs = [(j[0], 'text_log__' + column) if j[1] == column else j for j in _all_text_logs]
        if column in params:
            params = ['param__' + column if j == column else j for j in params]
            _all_params = [(j[0], 'param__' + column) if j[1] == column else j for j in _all_params]
        if column in properties:
            properties = ['property__' + column if j == column else j for j in properties]
            _all_properties = [(j[0], 'property__' + column) if j[1] == column else j for j in _all_properties]

    # Rename columns in DataFrame and sort experiments by neptune id
    new_col_names = {'id': 'neptune_id',
                     'owner': 'owner'}

    metrics = [(j[0], j[1]) for j in _all_metrics if j[1] in metrics]
    text_logs = [(j[0], j[1]) for j in _all_text_logs if j[1] in text_logs]
    params = [(j[0], j[1]) for j in _all_params if j[1] in params]
    properties = [(j[0], j[1]) for j in _all_properties if j[1] in properties]

    new_col_names.update(metrics)
    new_col_names.update(text_logs)
    new_col_names.update(params)
    new_col_names.update(properties)

    df = df[new_col_names.keys()]
    df = df.rename(columns=new_col_names)
    _exp_ids_series = df['neptune_id'].apply(lambda x: int(x.split('-')[-1]))
    df.insert(loc=0, column='neptune_exp_number', value=_exp_ids_series)
    df = df.astype({'neptune_exp_number': int})
    df = df.sort_values(by='neptune_exp_number', ascending=True)
    df = df.drop(columns='neptune_exp_number')

    # Prepare order of axes, where 'neptune_id' is first, metrics to the right.
    all_axes = df.columns.to_list()
    if metrics:
        metric_names = [j[1] for j in metrics]
        metric_names.sort()
        for metric in metric_names:
            all_axes.remove(metric)
        all_axes.sort()
        all_axes.sort(reverse=True, key='owner'.__eq__)
        all_axes.sort(reverse=True, key='neptune_id'.__eq__)
        all_axes = all_axes + metric_names

    # Prepare HiPlot visualization
    input_to_hiplot = df.T.to_dict().values()
    hiplot_vis = hip.Experiment().from_iterable(input_to_hiplot)
    for j, datapoint in enumerate(hiplot_vis.datapoints[1:], 1):
        datapoint.from_uid = hiplot_vis.datapoints[j - 1].uid

    # Save to html if requested
    if html_file_path is not None:
        assert isinstance(html_file_path, str), \
            '"html_file_path" should be string, but {} is given'.format(type(html_file_path))
        if os.path.dirname(html_file_path):
            os.makedirs(os.path.dirname(html_file_path), exist_ok=True)
        hiplot_vis.to_html(html_file_path)
    hiplot_vis.display_data(hip.Displays.PARALLEL_PLOT).update({'categoricalMaximumValues': df.shape[0],
                                                                'hide': ['uid', 'from_uid'],
                                                                'order': all_axes})
    return hiplot_vis.display()


def _validate_input(selected_columns, all_columns, type_name):
    all_columns = [j[1] for j in all_columns]
    if selected_columns is True:
        selected_columns = all_columns
    elif selected_columns is False:
        selected_columns = []
    elif isinstance(selected_columns, str):
        assert selected_columns in all_columns, \
            'There is no {} with a name "{}" in the project columns.'.format(type_name, selected_columns)
        selected_columns = [selected_columns, ]
    elif isinstance(selected_columns, list):
        for j in selected_columns:
            assert j in all_columns, 'There is no "{}" in the project columns.'.format(j)
    else:
        raise TypeError('{} must be None, string or list of strings'.format(selected_columns))
    return selected_columns
