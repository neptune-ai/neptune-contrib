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

import neptune
from neptunecontrib.monitoring.utils import fig2pil
from neptunecontrib.viz.utils import axes2fig
import numpy as np
import skopt.plots as sk_plots


class NeptuneMonitor:
    def __init__(self, ctx):
        self.ctx = ctx
        self.iteration = 0
        
    def __call__(self, res):
        self.ctx.channel_send('hyperparameter_search_score', 
                         x=self.iteration, y=res.func_vals[-1])
        self.ctx.channel_send('search_parameters', 
                         x=self.iteration, y=NeptuneMonitor._get_last_params(res))
        self.iteration+=1
    
    @staticmethod
    def _get_last_params(res):
        param_vals = res.x_iters[-1]
        named_params = _format_to_named_params(param_vals, res)
        return named_params


def send_best_parameters(results, ctx):
    param_vals = results.x
    named_params = _format_to_named_params(param_vals, results)
    ctx.channel_send('best_parameters', named_params)
    

def send_plot_convergence(results, ctx):
    convergence = fig2pil(axes2fig(sk_plots.plot_convergence(results)))
    ctx.channel_send('convergence', neptune.Image(
            name='convergence',
            description="plot_convergence from skopt",
            data=convergence))

    
def send_plot_evaluations(results, ctx):
    evaluations = fig2pil(axes2fig(sk_plots.plot_evaluations(results, bins=10)))
    ctx.channel_send('evaluations', neptune.Image(
            name='evaluations',
            description="plot_evaluations from skopt",
            data=evaluations))
    
    
def send_plot_objective(results, ctx):
    try:
        objective = fig2pil(axes2fig(sk_plots.plot_objective(results)))
        ctx.channel_send('objective', neptune.Image(
                name='objective',
                description="plot_objective from skopt",
                data=objective))
    except Exception as e:
        print('Could not create ans objective chart due to error: {}'.format(e))

                
def _format_to_named_params(params, result):
    param_names = [dim.name for dim in result.space.dimensions]
    named_params = []
    for name, val in zip(param_names, params):
        named_params.append((name, val))
    return named_params