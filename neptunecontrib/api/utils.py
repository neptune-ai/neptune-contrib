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

import pandas as pd


def concat_experiments_on_channel(experiments, channel_name):
    """Combines channel values from experiments into one dataframe.

    This function helps to compare channel values from a list of experiments
    by combining them in a dataframe. E.g: Say we want to extract the `log_loss`
    channel values for a list of experiments. The resulting dataframe will have
    ['id','x_log_loss','y_log_loss'] columns.

    Args:
        experiments(list): list of `neptunelib.api.Experiment` objects.
        channel_name(str): name of the channel for which we want to extract values.

    Returns:
        `pandas.DataFrame`: Dataframe of ['id','x_CHANNEL_NAME','y_CHANNEL_NAME']
        values concatenated from a list of experiments.

    Examples:
        Instantiate a session.

        >>> from neptunelib.api.session import Session
        >>> session = Session()

        Fetch a project and a list of experiments.

        >>> project = session.get_projects('neptune-ml')['neptune-ml/Salt-Detection']
        >>> experiments = project.get_experiments(state=['aborted'], owner=['neyo'], min_running_time=100000)

        Construct a channel value dataframe:

        >>> from neptunelib.api.utils import concat_experiments_on_channel
        >>> compare_df = concat_experiments_on_channel(experiments,'unet_0 epoch_val iout loss')

    Note:
        If an experiment in the list of experiments does not contain the channel with a specified channel_name
        it will be omitted.
    """
    combined_df = []
    for experiment in experiments:
        if channel_name in experiment.channels.keys():
            channel_df = experiment.get_numeric_channels_values(channel_name)
            channel_df['id'] = experiment.id
            combined_df.append(channel_df)
    combined_df = pd.concat(combined_df, axis=0)
    return combined_df


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
