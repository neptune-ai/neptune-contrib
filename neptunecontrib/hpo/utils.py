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

import subprocess
import random
import string

import neptune
import numpy as np
import skopt
import skopt.plots as sk_plots
from neptunelib.session import Session
from neptunecontrib.monitoring.utils import fig2pil
from neptunecontrib.viz.utils import axes2fig


def make_objective(param_set, command, metric_channel_name, project_name, tag='trash'):  
    command.insert(-1, '--tag {}'.format(tag))
    for name, value in param_set.items():
        command.insert(-1, "--parameter {}:{}".format(name, value))
    
    exp_id_tag = _get_random_string()
    command.insert(-1, '--tag {}'.format(exp_id_tag))
    
    command = " ".join(command)
    subprocess.call(command, shell=True)
    score = _get_score(exp_id_tag, metric_channel_name, project_name)    
    return score


class NeptuneMonitor:
    def __init__(self, ctx):
        self.ctx = ctx
        self.iteration = 0
        
    def __call__(self, res):
        self.ctx.channel_send('hyperparameter_search_score', 
                         x=self.iteration, y=res.func_vals[-1])
        self.ctx.channel_send('search_parameters', 
                         x=self.iteration, y=res.x_iters[-1])
        self.iteration+=1
        
        
class CheckpointSaver:
    """
    Save current state after each iteration with `skopt.dump`.
    Example usage:
        import skopt
        checkpoint_callback = skopt.callbacks.CheckpointSaver("./result.pkl")
        skopt.gp_minimize(obj_fun, dims, callback=[checkpoint_callback])
    Parameters
    ----------
    * `checkpoint_path`: location where checkpoint will be saved to;
    * `dump_options`: options to pass on to `skopt.dump`, like `compress=9`
    """
    def __init__(self, checkpoint_path, **dump_options):
        self.checkpoint_path = checkpoint_path
        self.dump_options = dump_options

    def __call__(self, res):
        """
        Parameters
        ----------
        * `res` [`OptimizeResult`, scipy object]:
            The optimization as a OptimizeResult object.
        """
        skopt.dump(res, self.checkpoint_path, **self.dump_options)


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
    except Exception:
        print('Could not create ans objective chart')
        
        
def _get_random_string():
    x = ''.join(random.choice(string.ascii_lowercase + string.digits) 
                for _ in range(64))
    return x 


def _get_score(exp_id_tag, metric_name, project_name):
    namespace = project_name.split('/')[0]
    
    session = Session()
    project = session.get_projects(namespace)[project_name]
    experiment = project.get_experiments(tag=[exp_id_tag])[0]
    score = float(experiment.properties[metric_name].tolist()[0])
    return score