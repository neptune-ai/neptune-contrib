#
# Copyright (c) 2019, Neptune Labs Sp. z o.o.
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

import altair as alt


def channel_curve_compare(experiment_df,
                          width=800,
                          heights=(50, 400),
                          line_size=5,
                          legend_mark_size=100):
    """Creates an interactive curve comparison chart for a list of experiments.

    It lets you tick or untick experiments that you want to compare by clicking on the legend (shift+click for multi),
    you can select the x range which you want to investigate by selecting it on the top chart and you
    get shown the actual values on mousehover.

    The chart is build on top of the Altair which in turn is build on top of Vega-Lite and Vega.
    That means you can use the objects produces by this script (converting it first to json by .to_json() method)
    in your html webpage without any problem.

    Args:
        experiment_df('pandas.DataFrame'): Dataframe containing ['id','x','CHANNEL_NAME'].
            It can be obtained from a list of experiments by using the
            `neptunelib.api.concat_experiments_on_channel` function. If the len of the dataframe exceeds 5000 it will
            cause the MaxRowsError. Read the Note to learn why and how to disable it.
        width(int): width of the chart. Default is 800.
        heights(tuple): heights of the subcharts. The first value controls the top chart, the second
            controls the bottom chart. Default is (50,400).
        line_size(int): size of the lines. Default is 5.
        legend_mark_size(int): size of the marks in legend. Default is 100.

    Returns:
        `altair.Chart`: Altair chart object which will be automatically rendered in the notebook. You can
        also run the `.to_json()` method on it to convert it to the Vega-Lite json format.

    Examples:
        Instantiate a session::

            from neptunelib.api.session import Session
            session = Session()

        Fetch a project and a list of experiments::

            project = session.get_projects('neptune-ml')['neptune-ml/Salt-Detection']
            experiments = project.get_experiments(state=['aborted'], owner=['neyo'], min_running_time=100000)

        Construct a channel value dataframe::

            from neptunelib.api.utils import concat_experiments_on_channel
            compare_df = concat_experiments_on_channel(experiments,'unet_0 epoch_val iout loss')

        Plot interactive chart in notebook::

            from neptunelib.viz.experiments import channel_curve_compare
            channel_curve_compare(compare_df)

    Note:
        Because Vega-Lite visualizations keep all the chart data in the HTML the visualizations can consume huge
        amounts of memory if not handled properly. That is why, by default the hard limit of 5000 rows is set to
        the len of dataframe. That being said, you can disable it by adding the following line in the notebook or code::

            import altair as alt
            alt.data_transformers.enable('default', max_rows=None)

    """

    assert len(experiment_df.columns) == 3, 'Experiment dataframe should have 3 columns \
        ["id","x", "CHANNEL_NAME"]. \
        It has {} namely {}'.format(len(experiment_df.columns), experiment_df.columns)

    top_height, bottom_height = heights
    prep_cols, channel_name = _preprocess_columns(experiment_df.columns)
    experiment_df.columns = prep_cols

    nearest = alt.selection(type='single', nearest=True, on='mouseover', fields=['x'], empty='none')
    interval = alt.selection(type='interval', encodings=['x'])
    legend_selection = alt.selection_multi(fields=['id'])

    legend = alt.Chart().mark_point(filled=True, size=legend_mark_size).encode(
        y=alt.Y('id:N'),
        color=alt.condition(legend_selection, alt.Color('id:N', legend=None), alt.value('lightgray'))
    ).add_selection(
        legend_selection
    )

    selectors = alt.Chart().mark_point().encode(
        x='x:Q',
        opacity=alt.value(0),
    ).add_selection(
        nearest
    )

    top_view = alt.Chart(width=width, height=top_height).mark_line(size=line_size).encode(
        x=alt.X('x:Q', title=None),
        y=alt.Y('y:Q', scale=alt.Scale(zero=False), title=None),
        color=alt.Color('id:N', legend=None),
        opacity=alt.condition(legend_selection, alt.OpacityValue(1), alt.OpacityValue(0.0))
    ).add_selection(
        interval
    )

    line = alt.Chart().mark_line(size=line_size).encode(
        x=alt.X('x:Q', title='iteration'),
        y=alt.Y('y:Q', scale=alt.Scale(zero=False), title=channel_name),
        color=alt.Color('id:N', legend=None),
        opacity=alt.condition(legend_selection, alt.OpacityValue(1), alt.OpacityValue(0.0))
    )

    points = line.mark_point().encode(
        color=alt.condition(legend_selection, alt.Color('id:N', legend=None), alt.value('white')),
        opacity=alt.condition(nearest, alt.value(1), alt.value(0))
    )

    text = line.mark_text(align='left', dx=5, dy=-5).encode(
        text=alt.condition(nearest, 'y:Q', alt.value(' ')),
        opacity=alt.condition(legend_selection, alt.OpacityValue(1), alt.OpacityValue(0.0))
    )

    rules = alt.Chart().mark_rule(color='gray').encode(
        x='x:Q',
    ).transform_filter(
        nearest
    )

    bottom_view = alt.layer(line, selectors, points, rules, text,
                            width=width, height=bottom_height
                           ).transform_filter(interval)

    combined = alt.hconcat(alt.vconcat(top_view, bottom_view), legend, data=experiment_df)
    return combined


def _preprocess_columns(columns):
    channel_name = _get_channel_name(columns)
    prep_cols = []
    for col in columns:
        if col == channel_name:
            col = 'y'
        prep_cols.append(col)
    return prep_cols, channel_name


def _get_channel_name(columns):
    return [col for col in columns if col not in ['id', 'x']][0]
