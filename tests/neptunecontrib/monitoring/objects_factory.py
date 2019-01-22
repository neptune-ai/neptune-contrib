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

import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.datasets import load_wine


def get_dategen():
    data = load_wine()
    cv = KFold(n_splits=3, random_state=1234)
    for train_idx, test_idx in cv.split(data.target):
        X_train, X_test = data.data[train_idx], data.data[test_idx]
        y_train, y_test = data.target[train_idx], data.target[test_idx]

        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

        yield lgb_train, lgb_eval
