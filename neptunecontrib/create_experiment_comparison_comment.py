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

"""Create a markdown file with an experiment comparison table.

Get a diff between experiments across metrics, parameters, and properties and save it to a file as a markdown table.

Attributes:
    experiment_ids(list(str)): Experiment ids of experiments you would like to compare. It works only for 2 experiments.
        You can pass it either as --experiment_ids or -e. For example, --experiment_ids GIT-83 GIT-82.
    tag_names(list(str)): tags of experiments you would like to compare.
        It works if tags passed are unique to the experiments they belong to.
        You can pass it either as --tag_names or -n. For example, --tag_names a892ee0ds 09asajd902.
    api_token(str): Neptune api token. If you have NEPTUNE_API_TOKEN environment
        variable set to your API token you can skip this parameter. You can pass it either as --neptune_api_token or -t.
    project_name(str): Full name of the project. E.g. "neptune-ai/neptune-examples",
        If you have PROJECT_NAME environment variable set to your Neptune project you can skip this parameter.
        You can pass it either as --project_name or -p.
    filepath(str): filepath of the output markdown file. You can pass it either as --filepath or -f.

Example:
    Create a file, comparison.md, with a comparison table of experiments GIT-83 and GIT-82::

        $ python -m neptunecontrib.create_experiment_comparison_comment \
            --experiment_ids GIT-83 GIT-82 \
            --api_token ANONYMOUS \
            --project_name shared/neptune-actions \
            --filepath comment_body.md

Note:
    If you keep your neptune api token in the NEPTUNE_API_TOKEN environment variable
    you can skip the --api_token.
    If you keep your full neptune project name in the PROJECT_NAME environment variable
    you can skip the --project_name.

"""

import argparse

import neptune
import numpy as np


def get_project(arguments):
    project = neptune.init(project_qualified_name=arguments.project_name,
                           api_token=arguments.api_token)
    return project


def get_experiment_data_by_id(arguments):
    project = get_project(arguments)
    return project.get_leaderboard(id=arguments.experiment_ids)


def get_experiment_data_by_tag(arguments):
    project = get_project(arguments)
    experiment_ids = [project.get_experiments(tag=tag)[0].id for tag in arguments.tag_names]

    assert len(experiment_ids) == 2, 'tags passed should be unique to the experiments they belong to'
    return project.get_leaderboard(id=experiment_ids)


def find_experiment_diff(df):
    selected_cols, cleaned_cols = [], []
    for col in df.columns:
        for name in ['channel_', 'parameter_', 'property_']:
            if name in col and not any(excluded_name in col for excluded_name in ['stderr', 'stdout']):
                selected_cols.append(col)
                cleaned_cols.append(col.replace(name, ''))
    df_selected = df[['id'] + selected_cols]

    different_cols = []
    for col in df_selected.columns:
        vals = df_selected[col].values
        if vals[0] != vals[1]:
            different_cols.append(col)

    return df_selected[different_cols]


def create_comment_markdown(df, project_name):
    # format data
    data = {'metrics': {},
            'parameters': {},
            'properties': {},
            'branches': ['main_branch', 'pull_request_branch']}

    df = df.iloc[::-1].reset_index()

    for k, v in df.to_dict().items():
        if k == 'id':
            data[k] = [v[0], v[1]]
        if 'channel_' in k:
            data['metrics'][k.replace('channel_', '')] = [v[0], v[1]]
        if 'parameter_' in k:
            data['parameters'][k.replace('parameter_', '')] = [v[0], v[1]]
        if 'property_' in k:
            data['properties'][k.replace('property_', '')] = [v[0], v[1]]

    user, project = project_name.split('/')

    # link to experiment comparison
    link = "https://ui.neptune.ai/{0}/{1}/compare?shortId=%5B%22{2}%22%2C%22{3}%22%5D".format(
        user, project, data['id'][0], data['id'][1])
    table = ["""<a href="{}">See the experiment comparison in Neptune </a>""".format(link)]
    table.append("<table><tr><td></td>")

    # add branch names section
    for branch in data['branches']:
        text = "<td><b>{}</b></td>".format(branch)
        table.append(text)

    # add experiment links and id section
    table.append("<tr><td>Neptune Experiment</td>")
    for exp_id in data['id']:
        text = """<td><a href="https://ui.neptune.ai/{0}/{1}/e/{2}"><b>{2}</b></a></td>""".format(user, project,
                                                                                                  exp_id)
        table.append(text)
    table.append("</tr>")

    # add metrics section
    if data['metrics']:
        table.append("""<tr>
            <th colspan=3, style="text-align:left;">
                Metrics
            </th>""")

        for name, values in data['metrics'].items():
            table.append("<tr><td>{}</td>".format(name))
            for value in values:
                try:
                    value = np.round(float(value), 5)
                except Exception:
                    pass
                table.append("<td>{}</td>".format(value))
            table.append("</tr>")

        table.append("</tr>")

    # add parameters section
    if data['parameters']:
        table.append("""<tr>
            <th colspan=3, style="text-align:left;">
                Parameters
            </th>""")

        for name, values in data['parameters'].items():
            table.append("<tr><td>{}</td>".format(name))
            for value in values:
                table.append("<td>{}</td>".format(value))
            table.append("</tr>")

        table.append("</tr>")

    # add properties sectio
    if data['properties']:
        table.append("""<tr>
            <th colspan=3, style="text-align:left;">
                Properties
            </th>""")

        for name, values in data['properties'].items():
            table.append("<tr><td>{}</td>".format(name))
            for value in values:
                table.append("<td>{}</td>".format(value))
            table.append("</tr>")

        table.append("</tr>")

    table.append("</tr></table>")

    table_text = "".join(table)

    return table_text


def main(arguments):
    if arguments.experiment_ids:
        df = get_experiment_data_by_id(arguments)
    elif arguments.tag_names:
        df = get_experiment_data_by_tag(arguments)
    else:
        ValueError("at least one of experiment_ids, tag_ids should be specified")

    df_diff = find_experiment_diff(df)
    comment_body = create_comment_markdown(df_diff, arguments.project_name)

    with open(arguments.filepath, "w+") as f:
        f.write(comment_body)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--tag_names', nargs=2, default=None)
    parser.add_argument('-e', '--experiment_ids', nargs=2, default=None)
    parser.add_argument('-t', '--api_token', default=None)
    parser.add_argument('-p', '--project_name', default=None)
    parser.add_argument('-f', '--filepath', default='comment.md')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
