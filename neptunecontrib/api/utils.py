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
import os
import tempfile
import warnings

import joblib
import neptune
import pandas as pd

warnings.filterwarnings('ignore')

__all__ = [
    'concat_experiments_on_channel',
    'extract_project_progress_info',
    'get_channel_columns',
    'get_parameter_columns',
    'get_property_columns',
    'get_system_columns',
    'strip_prefices',
    'log_pickle',
    'get_pickle'
]


def concat_experiments_on_channel(experiments, channel_name):
    """Combines channel values from experiments into one dataframe.

    This function helps to compare channel values from a list of experiments
    by combining them in a dataframe. E.g: Say we want to extract the `log_loss`
    channel values for a list of experiments. The resulting dataframe will have
    ['id','x_log_loss','y_log_loss'] columns.

    Args:
        experiments(list): list of `neptune.experiments.Experiment` objects.
        channel_name(str): name of the channel for which we want to extract values.

    Returns:
        `pandas.DataFrame`: Dataframe of ['id','x_CHANNEL_NAME','y_CHANNEL_NAME']
        values concatenated from a list of experiments.

    Examples:
        Instantiate a session::

            from neptune.sessions import Session
            session = Session()

        Fetch a project and a list of experiments::

            project = session.get_projects('neptune-ai')['neptune-ai/Salt-Detection']
            experiments = project.get_experiments(state=['aborted'], owner=['neyo'], min_running_time=100000)

        Construct a channel value dataframe::

            from neptunecontrib.api.utils import concat_experiments_on_channel
            compare_df = concat_experiments_on_channel(experiments,'unet_0 epoch_val iout loss')

    Note:
        If an experiment in the list of experiments does not contain the channel with a specified channel_name
        it will be omitted.
    """
    combined_df = []
    for experiment in experiments:
        if channel_name in experiment.get_channels().keys():
            channel_df = experiment.get_numeric_channels_values(channel_name)
            channel_df['id'] = experiment.id
            combined_df.append(channel_df)
    combined_df = pd.concat(combined_df, axis=0)
    return combined_df


def extract_project_progress_info(leadearboard, metric_colname,
                                  time_colname='finished'):
    """Extracts the project progress information from the experiment view.

    This function takes the experiment view (leaderboard) and extracts the information
    that is important for analysing the project progress. It creates additional columns
    `metric` (actual experiment metric), `metric_best` (best metric score to date)),
    `running_time_day` (total amount of experiment running time for a given day in hours),
    'experiment_count_day' (total number of experiments ran in a given day).

    This function is usually used with the `plot_project_progress` from `neptunecontrib.viz.projects`.

    Args:
        leadearboard(`pandas.DataFrame`): Dataframe containing the experiment view of the project.
            It can be extracted via `project.get_leaderboard()`.
        metric_colname(str): name of the column containing the metric of interest.
        time_colname(str): name of the column containing the timestamp. It can be either `finished`
            or `created`. Default is 'finished'.

    Returns:
        `pandas.DataFrame`: Dataframe of ['id', 'metric', 'metric_best', 'running_time',
        'running_time_day', 'experiment_count_day', 'owner', 'tags', 'timestamp', 'timestamp_day']
        columns.

    Examples:
        Instantiate a session::

            from neptune.sessions import Session
            session = Session()

        Fetch a project and the experiment view of that project::

            project = session.get_projects('neptune-ai')['neptune-ai/Salt-Detection']
            leaderboard = project.get_leaderboard()

        Create a progress info dataframe::

            from neptunecontrib.api.utils import extract_project_progress_info
            progress_df = extract_project_progress_info(leadearboard,
                                                        metric_colname='channel_IOUT',
                                                        time_colname='finished')
    """
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


def get_channel_columns(columns):
    """Filters leaderboard columns to get the channel column names.

    Args:
        columns(iterable): Iterable of leaderboard column names.

    Returns:
        list: A list of channel column names.
    """
    return [col for col in columns if col.startswith('channel_')]


def get_parameter_columns(columns):
    """Filters leaderboard columns to get the parameter column names.

    Args:
        columns(iterable): Iterable of leaderboard column names.

    Returns:
        list: A list of channel parameter names.
    """
    return [col for col in columns if col.startswith('parameter_')]


def get_property_columns(columns):
    """Filters leaderboard columns to get the property column names.

    Args:
        columns(iterable): Iterable of leaderboard column names.

    Returns:
        list: A list of channel property names.
    """
    return [col for col in columns if col.startswith('property_')]


def get_system_columns(columns):
    """Filters leaderboard columns to get the system column names.

    Args:
        columns(iterable): Iterable of leaderboard column names.

    Returns:
        list: A list of channel system names.
    """
    excluded_prefices = ['channel_', 'parameter_', 'property_']
    return [col for col in columns if not any([col.startswith(prefix) for
                                               prefix in excluded_prefices])]


def strip_prefices(columns, prefices):
    """Filters leaderboard columns to get the system column names.

    Args:
        columns(iterable): Iterable of leaderboard column names.
        prefices(list): List of prefices to strip. You can choose one of
            ['channel_', 'parameter_', 'property_']

    Returns:
        list: A list of clean column names.
    """
    new_columns = []
    for col in columns:
        for prefix in prefices:
            if col.startswith(prefix):
                col = col.replace(prefix, '')
        new_columns.append(col)
    return new_columns


def get_filepaths(dirpath='.', extensions=None):
    """Creates a list of all the files with selected extensions.

    Args:
        dirpath(str): Folder from which all files with given extensions should be added to list.
        extensions(list(str) or None): All extensions with which files should be added to the list.

    Returns:
        list: A list of filepaths with given extensions that are in the directory or subdirecotries.

    Examples:
        Initialize Neptune::

             import neptune
             from neptunecontrib.versioning.data import log_data_version
             neptune.init('USER_NAME/PROJECT_NAME')

        Create experiment and track all .py files from given directory and subdirs::

             with neptune.create_experiment(upload_source_files=get_filepaths(extensions=['.py'])):
                 neptune.send_metric('score', 0.97)

    """

    msg = """get_filepaths() is deprecated.
    Starting from neptune-client==4.9 you can pass ['**/*.py*', '**/*.yaml*', '**/*.yml*']
    to upload_source_files argument to upload all files with given extensions recursively.
    Read more https://docs.neptune.ai/neptune-client/docs/project.html
    get_filepaths() will be removed in future releases.
    """
    warnings.warn(msg, DeprecationWarning)

    if not extensions:
        extensions = ['.py', '.yaml', 'yml']
    files = []
    for r, _, f in os.walk(dirpath):
        for file in f:
            if any(file.endswith(ext) for ext in extensions):
                files.append(os.path.join(r, file))
    return files


def log_pickle(filename, obj, experiment=None):
    """Logs picklable object to Neptune.

    Pickles and logs your object to Neptune under specified filename.

    Args:
        obj: Picklable object.
        filename(str): filename under which object will be saved to Neptune.
        experiment(`neptune.experiments.Experiment`): Neptune experiment.

    Examples:
        Initialize Neptune::

            import neptune
            neptune.init('USER_NAME/PROJECT_NAME')

        Create RandomForest object and log to Neptune::

            from sklearn.ensemble import RandomForestClassifier
            from neptunecontrib.api import log_pickle

            neptune.create_experiment()

            rf = RandomForestClassifier()
            log_pickle('rf.pkl', rf)
    """
    _exp = experiment if experiment else neptune
    _exp.log_artifact(export_pickle(obj), filename)


def export_pickle(obj):
    from io import BytesIO
    import pickle

    buffer = BytesIO()
    pickle.dump(obj, buffer)
    buffer.seek(0)

    return buffer


def get_pickle(filename, experiment):
    """Downloads pickled artifact object from Neptune and returns a Python object.

    Downloads the pickled object from artifacts of given experiment,
     loads it to memory and returns a Python object.

    Args:
        filename(str): filename under which object will be saved to Neptune.
        experiment(`neptune.experiments.Experiment`): Neptune experiment.

    Examples:
        Initialize Neptune::

            import neptune

            project = neptune.init('USER_NAME/PROJECT_NAME')

        Choose Neptune experiment::

            experiment = project.get_experiments(id=['PRO-101'])[0]

        Get your pickled object from experiment artifacts::

            from neptunecontrib.api import get_pickle

            results = get_pickle('results.pkl', experiment)
    """
    with tempfile.TemporaryDirectory() as d:
        experiment.download_artifact(filename, d)
        full_path = os.path.join(d, filename)
        artifact = joblib.load(full_path)
    return artifact


def pickle_and_log_artifact(obj, filename, experiment=None):
    message = """pickle_and_log_artifact was renamed to log_pickle.
    You should use ``from neptunecontrib.api import log_pickle``
    neptunecontrib.api.pickle_and_log_artifact will be removed in future releases.
    """
    warnings.warn(message)

    log_pickle(filename, obj, experiment)


def get_pickled_artifact(experiment, filename):
    message = """get_pickled_artifact was renamed to get_pickle.
    You should use ``from neptunecontrib.api import get_pickle``
    neptunecontrib.api.get_pickled_artifact will be removed in future releases.
    """
    warnings.warn(message)

    get_pickle(filename, experiment)


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
    daily_counts = progress_df.groupby('timestamp_day').sum()[
        'running_time'].reset_index()
    daily_counts.columns = ['timestamp_day', 'running_time_day']
    progress_df = pd.merge(progress_df, daily_counts, on='timestamp_day')
    return progress_df


def _get_daily_experiment_counts(progress_df):
    daily_counts = progress_df.groupby('timestamp_day').count()[
        'metric'].reset_index()
    daily_counts.columns = ['timestamp_day', 'experiment_count_day']
    progress_df = pd.merge(progress_df, daily_counts, on='timestamp_day')
    return progress_df


def _get_current_best(progress_df):
    current_best = progress_df['metric'].cummax()
    current_best = current_best.fillna(method='bfill')
    progress_df['metric_best'] = current_best
    return progress_df
