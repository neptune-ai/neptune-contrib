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

from neptunelib.api.utils import map_keys, map_values, as_list


class TestMapValues(unittest.TestCase):
    def test_empty_map(self):
        # when
        mapped_dict = map_values(times_2, {})

        # then
        self.assertEqual({}, mapped_dict)

    def test_non_empty_map(self):
        # when
        mapped_dict = map_values(times_2, {'a': 2, 'b': 3})

        # then
        self.assertEqual({'a': 4, 'b': 6}, mapped_dict)


class TestMapKeys(unittest.TestCase):
    def test_empty_map(self):
        # when
        mapped_dict = map_keys(times_2, {})

        # then
        self.assertEqual({}, mapped_dict)

    def test_non_empty_map(self):
        # when
        mapped_dict = map_keys(times_2, {2: 'a', 3: 'b'})

        # then
        self.assertEqual({4: 'a', 6: 'b'}, mapped_dict)


class TestAsList(unittest.TestCase):

    def test_none(self):
        # expect
        self.assertEqual(None, as_list(None))

    def test_scalar(self):
        # expect
        self.assertEqual([1], as_list(1))

    def test_list(self):
        # expect
        self.assertEqual([2], as_list([2]))

    def test_dict(self):
        self.assertEqual([{'a': 1}], as_list({'a': 1}))


def times_2(x):
    return x * 2


if __name__ == '__main__':
    unittest.main()
