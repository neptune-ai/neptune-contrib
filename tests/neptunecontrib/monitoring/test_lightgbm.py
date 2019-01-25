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
import lightgbm as lgb

from neptunecontrib.monitoring.lightgbm import neptune_monitor
from tests.neptunecontrib.monitoring.objects_factory import get_dategen


class TestNeptuneMonitor(unittest.TestCase):
    def setUp(self):
        self.params = {'boosting_type': 'gbdt',
                       'objective': 'multiclass',
                       'num_class': 3,
                       'num_leaves': 5,
                       'learning_rate': 0.05,
                       'feature_fraction': 0.2
                       }
        
        self.ctx = neptune.Context()

    def test_k_fold_logging(self):
        self.ctx.reset_all_channels()

        for fold_id, (lgb_train, lgb_eval) in enumerate(get_dategen()):
            monitor = neptune_monitor(ctx=self.ctx, prefix='{}_'.format(fold_id))

            gbm = lgb.train(self.params,
                            lgb_train,
                            num_boost_round=3,
                            valid_sets=[lgb_train, lgb_eval],
                            valid_names=['train', 'valid'],
                            callbacks=[monitor],
                            )

        self.assertCountEqual(['0_train_multi_logloss',
                               '0_valid_multi_logloss',
                               '1_train_multi_logloss',
                               '1_valid_multi_logloss',
                               '2_train_multi_logloss',
                               '2_valid_multi_logloss'], self.ctx.job._channels.keys())

    def test_no_valid_names(self):
        self.ctx.reset_all_channels()

        for lgb_train, lgb_eval in get_dategen():
            monitor = neptune_monitor(ctx=self.ctx)

            gbm = lgb.train(self.params,
                            lgb_train,
                            num_boost_round=3,
                            valid_sets=[lgb_train, lgb_eval],
                            callbacks=[monitor],
                            )
            break

        self.assertCountEqual(['training_multi_logloss',
                               'valid_1_multi_logloss'], self.ctx.job._channels.keys())

    def test_logging_multiple_metrics(self):
        self.ctx.reset_all_channels()

        params = self.params
        params['metric'] = ['multi_logloss', 'multi_error']
        for lgb_train, lgb_eval in get_dategen():
            monitor = neptune_monitor(ctx=self.ctx)

            gbm = lgb.train(self.params,
                            lgb_train,
                            num_boost_round=3,
                            valid_sets=[lgb_train, lgb_eval],
                            valid_names=['train', 'valid'],
                            callbacks=[monitor],
                            )
            break

        self.assertCountEqual(['train_multi_error',
                               'train_multi_logloss',
                               'valid_multi_error',
                               'valid_multi_logloss'], self.ctx.job._channels.keys())


if __name__ == '__main__':
    unittest.main()
