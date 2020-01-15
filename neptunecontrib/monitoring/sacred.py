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

import collections
import warnings

import neptune
from neptunecontrib.monitoring.utils import pickle_and_send_artifact
from sacred.dependencies import get_digest
from sacred.observers import RunObserver


class NeptuneObserver(RunObserver):
    """Logs sacred experiment data to Neptune.

    Sacred observer that logs experiment metadata to neptune.
    The experiment data can be accessed and shared via web UI or experiment API.
    Check Neptune docs for more information https://docs.neptune.ai.

    Args:
        project_name(str): project name in Neptune app
        api_token(str): Neptune API token. If it is kept in the NEPTUNE_API_TOKEN environment
           variable leave None here.
        source_extensions(list(str)): list of extensions that Neptune should treat as source files
           extensions and send. If None is passed, Python file from which experiment was created will be uploaded.
           Pass empty list ([]) to upload no files. Unix style pathname pattern expansion is supported.
           For example, you can pass '*.py' to upload all python source files from the current directory.
           For recursion lookup use '**/*.py' (for Python 3.5 and later). For more information see glob library.

    Examples:
        Create sacred experiment::

            from numpy.random import permutation
            from sklearn import svm, datasets
            from sacred import Experiment

            ex = Experiment('iris_rbf_svm')

        Add Neptune observer::

            from neptunecontrib.monitoring.sacred import NeptuneObserver
            ex.observers.append(NeptuneObserver(api_token='YOUR_LONG_API_TOKEN',
                                                project_name='USER_NAME/PROJECT_NAME'))

        Run experiment::

            @ex.config
            def cfg():
                C = 1.0
                gamma = 0.7

            @ex.automain
            def run(C, gamma, _run):
                iris = datasets.load_iris()
                per = permutation(iris.target.size)
                iris.data = iris.data[per]
                iris.target = iris.target[per]
                clf = svm.SVC(C, 'rbf', gamma=gamma)
                clf.fit(iris.data[:90],
                        iris.target[:90])
                return clf.score(iris.data[90:],
                                 iris.target[90:])

        Go to the app and see the experiment. For example, https://ui.neptune.ai/jakub-czakon/examples/e/EX-341
    """

    def __init__(self, project_name, api_token=None, source_extensions=None):
        neptune.init(project_qualified_name=project_name, api_token=api_token)

        self.resources = {}

        if source_extensions:
            self.source_extensions = source_extensions
        else:
            self.source_extensions = ['**/*.py', '**/*.yaml', '**/*.yml']

    def started_event(self, ex_info, command, host_info, start_time, config, meta_info, _id):

        neptune.create_experiment(name=ex_info['name'],
                                  params=_flatten_dict(config),
                                  upload_source_files=self.source_extensions,
                                  properties={'mainfile': ex_info['mainfile'],
                                              'dependencies': str(ex_info['dependencies']),
                                              'sacred_id': str(_id),
                                              **_str_dict_values(host_info),
                                              **_str_dict_values(_flatten_dict(meta_info)),
                                              **_str_dict_values(_flatten_dict(ex_info))},
                                  git_info=neptune.utils.get_git_info(ex_info['base_dir']))

    def completed_event(self, stop_time, result):
        if result:
            if not isinstance(result, tuple):
                result = (
                    result,)  # transform single result to tuple so that both single & multiple results use same code

            for i, r in enumerate(result):
                if isinstance(r, float) or isinstance(r, int):
                    neptune.log_metric("result_{}".format(i), float(r))
                elif isinstance(r, object):
                    pickle_and_send_artifact(r, "result_{}.pkl".format(i))
                else:
                    warnings.warn(
                        "logging results does not support type '{}' results. Ignoring this result".format(type(r)))

        neptune.stop()

    def interrupted_event(self, interrupt_time, status):
        neptune.stop()

    def failed_event(self, fail_time, fail_trace):
        neptune.stop()

    def artifact_event(self, name, filename, metadata=None, content_type=None):
        neptune.log_artifact(filename)

    def resource_event(self, filename):
        if filename not in self.resources:
            md5 = get_digest(filename)
            self.resources[filename] = md5

        neptune.set_property('resources', str(list(self.resources.keys())))
        neptune.set_property(filename, self.resources[filename])

    def log_metrics(self, metrics_by_name, info):
        for metric_name, metric_ptr in metrics_by_name.items():
            for step, value in zip(metric_ptr["steps"], metric_ptr["values"]):
                neptune.log_metric(metric_name, x=step, y=value)


def _flatten_dict(d, parent_key='', sep=' '):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def _str_dict_values(d):
    return {k: str(v) for k, v in d.items()}
