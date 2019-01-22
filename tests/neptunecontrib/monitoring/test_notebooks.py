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

import unittest

import neptune
import numpy as np

from neptunecontrib.monitoring.notebooks import LocalNotebookContext


class TestLocalNotebookContext(unittest.TestCase):

    def test_get_params(self):
        # when
        ctx = neptune.Context()
        ctx = LocalNotebookContext(ctx, config_filepath='tests/neptunelib/monitoring/test_config.yaml')

        # then
        self.assertEqual({'lr': 0.01, 'model': 'resnet18'}, ctx.params)

    def test_x_none(self):
        # when
        ctx = neptune.Context()
        ctx = LocalNotebookContext(ctx, config_filepath='tests/neptunelib/monitoring/test_config.yaml')
        for i in range(10):
            ctx.channel_send('test_channel', y=np.random.random())

        # then
        self.assertEqual(list(range(10)), ctx.numeric_channels['test_channel'].x)


if __name__ == '__main__':
    unittest.main()
