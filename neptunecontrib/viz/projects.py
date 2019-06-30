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
import warnings

import altair as alt
import pandas as pd

warnings.filterwarnings('ignore')


def project_progress(progress_df,
                     width=800,
                     heights=(50, 400),
                     line_size=5,
                     text_size=15,
                     opacity=0.3):
    """Creates an interactive project progress exploration chart.

    It lets you choose the resources you want to see ('experiment_count_day' or 'running_time_day'), you
    can see the metric/id/tags for every experiment on mouseover, you can select the x range which you want to
    investigate by selecting it on the top chart and you get shown the actual values on mousehover.

    The chart is build on top of the Altair which in turn is build on top of Vega-Lite and Vega.
    That means you can use the objects produces by this script (converting it first to json by .to_json() method)
    in your html webpage without any problem.

    Args:
        progress_df('pandas.DataFrame'): Dataframe containing ['id', 'metric', 'metric_best', 'running_time',
            'running_time_day', 'experiment_count_day', 'owner', 'tags', 'timestamp', 'timestamp_day'].
            It can be obtained from a list of experiments by using the
            `neptunecontrib.api.extract_project_progress_info` function.
            If the len of the dataframe exceeds 5000 it will cause the MaxRowsError.
            Read the Note to learn why and how to disable it.
        width(int): width of the chart. Default is 800.
        heights(tuple): heights of the subcharts. The first value controls the top chart, the second
            controls the bottom chart. Default is (50,400).
        line_size(int): size of the lines. Default is 5.
        text_size(int): size of the text containing metric/id/tags in the middle.
        opacity(float): opacity of the resource bars in the background. Default is 0.3.

    Returns:
        `altair.Chart`: Altair chart object which will be automatically rendered in the notebook. You can
        also run the `.to_json()` method on it to convert it to the Vega-Lite json format.

    Examples:
        Instantiate a session::

            from neptunelib.api.session import Session
            session = Session()

        Fetch a project and the experiment view of that project::

            project = session.get_projects('neptune-ml')['neptune-ml/Salt-Detection']
            leaderboard = project.get_leaderboard()

        Create a progress info dataframe::

            from neptunecontrib.api.utils import extract_project_progress_info
            progress_df = extract_project_progress_info(leadearboard,
                                                        metric_colname='channel_IOUT',
                                                        time_colname='finished')

        Plot interactive chart in notebook::

            from neptunecontrib.viz.projects import project_progress
            project_progress(progress_df)

    Note:
        Because Vega-Lite visualizations keep all the chart data in the HTML the visualizations can consume huge
        amounts of memory if not handled properly. That is why, by default the hard limit of 5000 rows is set to
        the len of dataframe. That being said, you can disable it by adding the following line in the notebook or code::

            import altair as alt
            alt.data_transformers.enable('default', max_rows=None)

    """
    top_height, bottom_height = heights

    progress_df = _prep_progress_df(progress_df)

    nearest = alt.selection(type='single', nearest=True, on='mouseover',
                            fields=['timestamp'], empty='none')
    brush = alt.selection(type='interval', encodings=['x'])
    exp_box = alt.binding_select(options=['running_time_day', 'experiment_count_day'])
    exp_selection = alt.selection_single(name='select', fields=['resource'], bind=exp_box)

    top_view = alt.Chart(height=top_height, width=width).mark_line(interpolate='step-after', size=line_size).encode(
        x='timestamp:T',
        y=alt.Y('metric:Q', scale=alt.Scale(zero=False), axis=None),
        color=alt.Color('actual_or_best:N', legend=alt.Legend(title='Metric actual or current best')),
    ).add_selection(
        brush
    )

    selectors = alt.Chart().mark_point().encode(
        x=alt.X('timestamp:T'),
        opacity=alt.value(0),
    ).add_selection(
        nearest
    ).transform_filter(
        brush
    )
    line = alt.Chart().mark_line(interpolate='step-after', size=line_size).encode(
        x=alt.X('timestamp:T'),
        y=alt.Y('metric:Q', scale=alt.Scale(zero=False)),
        color=alt.Color('actual_or_best:N', legend=alt.Legend(title='Metric actual or current best')),
    ).transform_filter(
        brush
    )
    points = line.mark_point().encode(
        opacity=alt.condition(nearest, alt.value(1), alt.value(0))
    )
    text = line.mark_text(align='left', dx=5, dy=-5, size=text_size).encode(
        text=alt.condition(nearest, 'metric:Q', alt.value(' ')),
        color='actual_or_best:N'
    )
    rules = alt.Chart().mark_rule(color='gray').encode(
        x=alt.X('timestamp:T'),
    ).transform_filter(
        nearest
    )
    metrics = alt.layer(line, points, text, rules, selectors).properties(
        height=bottom_height,
        width=width,
    )

    exp_selector = alt.Chart().mark_area().encode(
        x=alt.X('timestamp:T'),
        opacity=alt.value(0),
    ).add_selection(
        exp_selection
    ).transform_filter(
        exp_selection
    ).transform_filter(
        brush
    )
    exp_line = alt.Chart().mark_area(interpolate='step-after').encode(
        x=alt.X('timestamp:T'),
        y=alt.Y('time_or_count:Q', scale=alt.Scale(zero=False)),
        color=alt.ColorValue('red'),
        opacity=alt.OpacityValue(opacity)
    ).transform_filter(
        brush
    ).transform_filter(
        exp_selection
    )
    exp_points = exp_line.mark_point(filled=True).encode(
        color=alt.ColorValue('black'),
        opacity=alt.condition(nearest, alt.value(1), alt.value(0))
    )
    exp_text = exp_line.mark_text(align='left', dx=5, dy=-5, fontWeight='bold', size=text_size).encode(
        text=alt.condition(nearest, 'time_or_count:Q', alt.value(' ')),
        color=alt.ColorValue('black')
    )
    exp_rules = alt.Chart().mark_rule(color='gray').encode(
        x=alt.X('timestamp:T'),
    ).transform_filter(
        nearest
    )
    exps = alt.layer(exp_line, exp_points, exp_rules, exp_text, exp_selector).properties(
        height=bottom_height,
        width=width,
    )

    main_view = alt.layer(exps, metrics).properties(
        height=bottom_height,
        width=width,
    ).resolve_scale(
        y='independent'
    )

    tags = alt.Chart(height=1, width=1).mark_text(align='left', size=text_size, fontWeight='bold').encode(
        x=alt.X('timestamp:T', axis=None),
        text=alt.condition(nearest, 'text:N', alt.value(' ')),
    )

    combined = alt.vconcat(top_view, tags, main_view, data=progress_df)
    return combined


def _prep_progress_df(progress_df):
    progress_df['text'] = progress_df.apply(_get_text, axis=1)

    metric_df = progress_df[['id', 'metric', 'metric_best']]
    metric_df.columns = ['id', 'actual', 'best']
    metric_df = metric_df.melt(id_vars=['id'],
                               value_vars=['actual', 'best'],
                               var_name='actual_or_best',
                               value_name='metric')

    exp_df = progress_df[['id', 'experiment_count_day', 'running_time_day']]
    exp_df.columns = ['id', 'experiment_count_day', 'running_time_day']
    exp_df['running_time_day'] = exp_df['running_time_day'] / (60 * 60)
    exp_df = exp_df.melt(id_vars=['id'],
                         value_vars=['experiment_count_day', 'running_time_day'],
                         var_name='resource',
                         value_name='time_or_count')

    progress_df = progress_df.drop(labels=['metric', 'metric_best', 'experiment_count_day', 'running_time_day'], axis=1)
    progress_df = pd.merge(metric_df, progress_df, on='id')
    progress_df = pd.merge(exp_df, progress_df, on='id')

    progress_df['timestamp'] = progress_df['timestamp'].astype(str)
    progress_df['timestamp_day'] = progress_df['timestamp_day'].astype(str)
    return progress_df


def _get_text(row):
    text = '{0} | {1:.4f} | {2}'.format(row['id'],
                                        row['metric'],
                                        ' ({})'.format(' , '.join(row['tags'])))
    return text
