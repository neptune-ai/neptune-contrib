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

import argparse
import json
from subprocess import call


def write_main_content(main_file, experiment_filepath):
    main_content = """
import neptune
import json

ctx = neptune.Context()
    
with open('{}', 'r') as fp:
    data = json.load(fp)
        
for name, channel in data['channels'].items():
    for x, y in zip(channel['x'], channel['y']):
        ctx.channel_send(name, x, y)
    
for name, value in data['properties'].items():
    ctx.properties[name] = value
            
ctx.tags.extend(data['tags'])
"""
    
    main_file.write(main_content.format(experiment_filepath))
    
    
def write_config_content(config_file, experiment_filepath):
    with open(experiment_filepath, 'r') as fp:
        data = json.load(fp)
    
    config_file.write('name: {}\n\n'.format(data['name']))
    config_file.write('parameters:\n')
    for name, value in data['parameters'].items():
        config_file.write('   {}: {}\n'.format(name, value))
    
    
def main(args):
    with open('neptune_sync_main.py', 'w') as main:
        write_main_content(main, args.filepath)
        
    with open('neptune_sync_config.yaml', 'w') as config:
        write_config_content(config, args.filepath)
            
    call('neptune run \
        --exclude neptune_sync_main.py \
        --project {} \
        --config neptune_sync_config.yaml \
        neptune_sync_main.py'.format(args.project_name), shell=True)
    call('rm neptune_sync_config.yaml neptune_sync_main.py', shell=True)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filepath')
    parser.add_argument('-p', '--project_name')
    args = parser.parse_args()
    
    main(args)
    