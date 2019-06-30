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

"""Syncs json file containg experiment data with Neptune project.

You can run your experiment in any language, create a `.json` file
that contains your hyper parameters, metrics, tags or properties and send that to Neptune.

Attributes:
    filepath(str): filepath to the `.json` file that contains experiment data. It can have
        ['tags', 'channels', 'properties', 'parameters', 'name'] sections.
        You can pass it either as --filepath or -f.
    project_name(str): Full name of the project. E.g. "neptune-ml/neptune-examples",
        You can pass it either as --project_name or -p.
    neptune_api_token(str): Neptune api token. If you have NEPTUNE_API_TOKEN environment
        variable set to your API token you can skip this parameter.
        You can pass it either as --neptune_api_token or -t defaults to NEPTUNE_API_TOKEN.

Example:
    Run the experiment and create experiment json in any language.
    For example, lets say your `experiment_data.json` is::

        {
        'name': 'example',
        'description': 'json tracking experiment',
        'params': {'lr': 0.1,
                   'batch_size': 128,
                   'dropount': 0.5
                   },
        'properties': {'data_version': '1231ffwefef9',
                       'data_path': '/mnt/path/to/data'
                       },
        'tags': ['resnet', 'no_preprocessing'],
        'upload_source_files': ['train.sh'],
        'send_metric': {'log_loss': {'x': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                                     'y': [0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100]
                                     },
                        'accuracy': {'x': [0, 4, 5, 8, 9],
                                     'y': [0.23, 0.47, 0.62, 0.89, 0.92]
                                     }
                        },
        'send_text': {'hash': {'x': [0, 4, 5, 8, 9],
                               'y': ['123123', 'as32e132', '123sdads', '123asdasd', ' asd324132a']
                               },
                      },
        'send_image': {'heatmaps': {'x': [0, 1, 2],
                                    'y': ['img1.png', 'img2.png', 'img3.png']
                                    },
                       },
        }

    Now you can sync your file with neptune::

        $ python neptunecontrib.sync.with_json
            --neptune_api_token 'ey7123qwwskdnaqsojnd1ru0129e12e=='
            --project_name neptune-ml/neptune-examples
            --filepath experiment_data.json

Note:
    If you keep your neptune api token in the NEPTUNE_API_TOKEN environment variable
    you can skip the --neptune_api_token

"""

import argparse
import json

import neptune


def main(arguments):
    with open(arguments.filepath, 'r') as fp:
        json_exp = json.load(fp)

    neptune.init(api_token=arguments.neptune_api_token,
                 project_qualified_name=arguments.project_name)

    with neptune.create_experiment(name=json_exp['name'],
                                   description=json_exp['description'],
                                   params=json_exp['params'],
                                   properties=json_exp['properties'],
                                   tags=json_exp['tags'],
                                   upload_source_files=json_exp['upload_source_files']):

        for name, channel_xy in json_exp['send_metric'].items():
            for x, y in zip(channel_xy['x'], channel_xy['y']):
                neptune.send_metric(name, x=x, y=y)

        for name, channel_xy in json_exp['send_text'].items():
            for x, y in zip(channel_xy['x'], channel_xy['y']):
                neptune.send_text(name, x=x, y=y)

        for name, channel_xy in json_exp['send_image'].items():
            for x, y in zip(channel_xy['x'], channel_xy['y']):
                neptune.send_image(name, x=x, y=y)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filepath')
    parser.add_argument('-t', '--neptune_api_token', default=None)
    parser.add_argument('-p', '--project_name')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
