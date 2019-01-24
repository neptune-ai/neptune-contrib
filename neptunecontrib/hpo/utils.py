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
from neptunelib.session import Session
        
        
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
        
        
def _get_random_string(length=64):
    x = ''.join(random.choice(string.ascii_lowercase + string.digits) 
                for _ in range(length))
    return x 


def _get_score(exp_id_tag, metric_name, project_name):
    namespace = project_name.split('/')[0]
    
    session = Session()
    project = session.get_projects(namespace)[project_name]
    experiment = project.get_experiments(tag=[exp_id_tag])[0]
    score = float(experiment.properties[metric_name].tolist()[0])
    return score