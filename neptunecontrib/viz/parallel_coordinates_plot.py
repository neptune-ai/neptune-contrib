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
import hiplot as hip

def make_parallel_coordinates_plot(columns=None, id=None, state=None, owner=None, tag=None, min_running_time=None):
    """

    Args:
        columns: If user leave None, then all parameters, id and owner are displayed.
        id:
        state:
        owner:
        tag:
        min_running_time:

    Returns:

    """
    if neptune.project is None:
        msg = """You do not have project, from which to fetch data.
                 Use neptune.init() to set project, for example: neptune.init('USERNAME/example-project').
                 See docs: https://docs.neptune.ai/neptune-client/docs/neptune.html#neptune.init"""
        raise ValueError(msg)

    df = neptune.project.get_leaderboard(id=id, state=state, owner=owner, tag=tag, min_running_time=min_running_time)

    if len(df) == 0:
        exit_msg = 'No experiments to show. Try other filters.'
    else:
        if columns is None:
            columns = ['id', 'owner']
            for col_name in df.columns.to_list():
                if 'parameter_' in col_name:
                    columns.append(col_name)
        elif not all(column in df.columns.to_list() for column in columns):
            raise ValueError('Check "columns" parameter, for columns that are not in this project.')
        df = df[columns]
        input_to_hiplot = df.T.to_dict().values()
        hip.Experiment.from_iterable(input_to_hiplot).display()
