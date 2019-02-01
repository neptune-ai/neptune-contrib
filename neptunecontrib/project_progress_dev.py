import altair as alt
import pandas as pd


def create_progress_df(leadearboard, metric_colname, time_colname):
    system_columns = ['id', 'owner', 'running_time', 'tags']
    progress_columns = system_columns + [time_colname, metric_colname]
    progress_df = leadearboard[progress_columns]
    progress_df.columns = ['id', 'owner', 'running_time', 'tags'] + ['timestamp', 'metric']

    progress_df = _prep_time_column(progress_df)
    progress_df = _prep_metric_column(progress_df)

    progress_df = _get_daily_running_time(progress_df)
    progress_df = _get_daily_experiment_counts(progress_df)
    progress_df = _get_current_best(progress_df)
    progress_df = progress_df[
        ['id', 'metric', 'metric_best', 'running_time', 'running_time_day', 'experiment_count_day',
         'owner', 'tags', 'timestamp', 'timestamp_day']]

    return progress_df


def plot_project_progress(progress_df,
                          width=800,
                          heights=(50, 400),
                          line_size=5,
                          text_size=15):
    top_height, bottom_height = heights

    progress_df = _prep_progress_df(progress_df)

    nearest = alt.selection(type='single', nearest=True, on='mouseover',
                            fields=['timestamp'], empty='none')

    brush = alt.selection(type='interval', encodings=['x'])

    top_view = alt.Chart(height=top_height, width=width).mark_line(interpolate='step-after', size=line_size).encode(
        x='timestamp:T',
        y=alt.Y('metric:Q', scale=alt.Scale(zero=False), axis=None),
        color='actual_or_best:N'
    ).add_selection(
        brush
    )

    selectors = alt.Chart().mark_point().encode(
        x=alt.X('timestamp:T', scale=alt.Scale()),
        opacity=alt.value(0),
    ).add_selection(
        nearest
    ).transform_filter(
        brush
    )

    line = alt.Chart().mark_line(interpolate='step-after', size=line_size).encode(
        x=alt.X('timestamp:T', scale=alt.Scale()),
        y=alt.Y('metric:Q', scale=alt.Scale(zero=False)),
        color='actual_or_best:N'
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
        x=alt.X('timestamp:T', scale=alt.Scale()),
    ).transform_filter(
        nearest
    )

    metrics = alt.layer(line, points, text, rules, selectors).properties(
        height=bottom_height,
        width=width,
    )

    exp_box = alt.binding_select(options=['running_time_day', 'count_day'])
    exp_selection = alt.selection_single(name='experiment', fields=['time_count'], bind=exp_box)

    exp_line = alt.Chart().mark_area(interpolate='step-after', size=line_size).encode(
        x=alt.X('timestamp:T', scale=alt.Scale()),
        y=alt.Y('time_count_value:Q', scale=alt.Scale(zero=False)),
        color=alt.ColorValue('pink'),
        opacity=alt.OpacityValue(0.5)
    ).transform_filter(
        brush
    ).add_selection(
        exp_selection
    ).transform_filter(
        exp_selection
    )

    exp_points = exp_line.mark_point(filled=True).encode(
        color=alt.ColorValue('black'),
        opacity=alt.condition(nearest, alt.value(1), alt.value(0))
    )

    exp_text = exp_line.mark_text(align='left', dx=5, dy=-5, fontWeight='bold', size=text_size).encode(
        text=alt.condition(nearest, 'time_count_value:Q', alt.value(' ')),
        color=alt.ColorValue('black')
    )

    exp_rules = alt.Chart().mark_rule(color='gray').encode(
        x=alt.X('finished:T', scale=alt.Scale()),
    ).transform_filter(
        nearest
    )

    exps = alt.layer(exp_line, exp_points, exp_rules, exp_text).properties(
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


def _prep_time_column(progress_df):
    progress_df['timestamp'] = pd.to_datetime(progress_df['timestamp'])
    progress_df.sort_values('timestamp', inplace=True)
    progress_df['timestamp_day'] = [d.date() for d in progress_df['timestamp']]
    return progress_df


def _prep_metric_column(progress_df):
    progress_df['metric'] = progress_df['metric'].astype(float)
    progress_df.dropna(subset=['metric'], how='all', inplace=True)
    return progress_df


def _get_daily_running_time(progress_df):
    daily_counts = progress_df.groupby('timestamp_day').sum()['running_time'].reset_index()
    daily_counts.columns = ['timestamp_day', 'running_time_day']
    progress_df = pd.merge(progress_df, daily_counts, on='timestamp_day')
    return progress_df


def _get_daily_experiment_counts(progress_df):
    daily_counts = progress_df.groupby('timestamp_day').count()['metric'].reset_index()
    daily_counts.columns = ['timestamp_day', 'experiment_count_day']
    progress_df = pd.merge(progress_df, daily_counts, on='timestamp_day')
    return progress_df


def _get_current_best(progress_df):
    current_best = progress_df['metric'].cummax()
    current_best = current_best.fillna(method='bfill')
    progress_df['metric_best'] = current_best
    return progress_df


def _prep_progress_df(progress_df):
    progress_df['text'] = progress_df.apply(_get_text, axis=1)

    metric_df = progress_df[['id', 'metric', 'metric_best']]
    metric_df.columns = ['id', 'actual', 'best']
    metric_df = metric_df.melt(id_vars=['id'],
                               value_vars=['actual', 'best'],
                               var_name='actual_or_best',
                               value_name='metric'
                               )

    exp_df = progress_df[['id', 'experiment_count_day', 'running_time_day']]
    exp_df.columns = ['id', 'experiment_count_day', 'running_time_day']
    exp_df = exp_df.melt(id_vars=['id'],
                         value_vars=['experiment_count_day', 'running_time_day'],
                         var_name='time_count',
                         value_name='time_count_value'
                         )

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
