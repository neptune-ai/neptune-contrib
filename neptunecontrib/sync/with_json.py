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
that contains your hyper parameters, metrics, tags or properties and log that to Neptune.

Attributes:
    filepath(str): filepath to the `.json` file that contains experiment data. It can have
        ['tags', 'channels', 'properties', 'parameters', 'name', 'log_metric', 'log_image', 'log_artifact'] sections.
        You can pass it either as --filepath or -f.
    project_name(str): Full name of the project. E.g. "neptune-ai/neptune-examples",
        If you have PROJECT_NAME environment variable set to your Neptune project you can skip this parameter.
        You can pass it either as --project_name or -p.
    neptune_api_token(str): Neptune api token. If you have NEPTUNE_API_TOKEN environment
        variable set to your API token you can skip this parameter.
        You can pass it either as --neptune_api_token or -t.

Example:
    Run the experiment and create experiment json in any language.
    For example, lets say your `experiment_data.json` is::

        {
          "name": "example",
          "description": "json tracking experiment",
          "params": {
            "lr": 0.1,
            "batch_size": 128,
            "dropount": 0.5
          },
          "properties": {
            "data_version": "1231ffwefef9",
            "data_path": "data/train.csv"
          },
          "tags": [
            "resnet",
            "no_preprocessing"
          ],
          "upload_source_files": [
            "run.sh"
          ],
          "log_metric": {
            "log_loss": {
              "x": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
              "y": [0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100]
            },
            "accuracy": {
              "x": [0, 4, 5, 8, 9],
              "y": [0.23, 0.47, 0.62, 0.89, 0.92]
            }
          },
          "log_text": {
            "hash": {
              "x": [0, 4, 5, 8, 9],
              "y": ["123123", "as32e132", "123sdads", "123asdasd", " asd324132a"]
            }
          },
          "log_image": {
            "diagnostic_charts": {
              "x": [0, 1, 2],
              "y": ["data/roc_auc_curve.png", "data/confusion_matrix.png"
              ]
            }
          },
          "log_artifact": ["data/model.pkl", "data/results.csv"]
        }


    Now you can sync your file with neptune::

        $ python neptunecontrib.sync.with_json
            --neptune_api_token 'ey7123qwwskdnaqsojnd1ru0129e12e=='
            --project_name neptune-ai/neptune-examples
            --filepath experiment_data.json

Checkout an example experiment here:
https://ui.neptune.ai/o/shared/org/any-language-integration/e/AN-2/logs

Note:
    If you keep your neptune api token in the NEPTUNE_API_TOKEN environment variable
    you can skip the --neptune_api_token

"""
import warnings

from neptunecontrib.create_experiment_from_json import main, parse_args

message = """neptunecontrib.logging.chart was moved to neptunecontrib.api.
You should use ``from neptunecontrib.api import log_chart`` 
neptunecontrib.logging.log_chart will be removed in future releases.
"""
warnings.warn(message)

if __name__ == '__main__':
    args = parse_args()
    main(args)
