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

import numpy as np
import matplotlib.pyplot as plt
from PIL.Image import Image
import seaborn as sns

from neptunecontrib.monitoring.utils import fig2pil, is_offline_context


class TestFig2Pil(unittest.TestCase):
    def test_matplotlib_figure(self):
        # when
        figure = plt.figure()
        pil_figure = fig2pil(figure)

        # then
        self.assertIsInstance(pil_figure, Image)

    def test_seaborn_distplot(self):
        # when
        figure = plt.figure()
        sns.distplot(np.random.random(100))
        pil_figure = fig2pil(figure)

        # then
        self.assertIsInstance(pil_figure, Image)

        
class TestIsOfflineContext(unittest.TestCase):
    def test_dummy(self):
        # when

        # then
        self.assertEqual({}, {})
        
if __name__ == '__main__':
    unittest.main()
