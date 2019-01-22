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

import os

import yaml
from attrdict import AttrDict
import matplotlib.pyplot as plt
from IPython.display import clear_output

from neptunecontrib.monitoring.utils import is_offline_context


def LocalNotebookContext(context, config_filepath, figsize=(16, 12), max_cols=2):
    """Wrapper for the Neptune context that lets you work in local notebooks.

    This wrapper makes it possible to access parameters and plot numerical channels
    to the notebook without any modifications to your code. During the development phase
    you can simply work in notebook and explore learning curves just as you would do in
    the Neptune-app.

    Args:
        context('neptune.Context'): Neptune context.
        config_filepath(str): filepath to the Neptune configuration file.
        figsize(tuple): The size of the figure on where the numerical channels are plotted.
            Default is (16,12).
        max_cols(int): Number of charts that should be plotted in one row. Default is 2.

    Examples:
        Create Neptune context instance:

        >>> import neptune
        >>> ctx = neptune.Context()

        Create Local Neptune context:

        >>> ctx = LocalNotebookContext(ctx, config_filepath='neptune.yaml')

        Access parameters from code:

        >>> ctx.params

        Send values to numerical channels and plot:

        >>> ctx.channel_send('loss', x=1, y=0.92)

    Note:
        After the development process is over you run your notebook as a
        python script and send the metrics to the Neptune app.
        Simply run the following command.

        $ jupyter nbconvert --to script main.py; neptune run --config neptune.yaml main.py
    """
    if is_offline_context(context):
        return LocalPatchedContext(context=context, config_filepath=config_filepath, figsize=figsize, max_cols=max_cols)
    else:
        return context


class LocalPatchedContext(object):
    """Wrapper for the Neptune context that lets you work in local notebooks.

    This wrapper makes it possible to access parameters and plot numerical channels
    to the notebook without any modifications to your code. During the development phase
    you can simply work in notebook and explore learning curves just as you would do in
    the Neptune-app.

    Args:
        context('neptune.Context'): Neptune context.
        config_filepath(str): filepath to the Neptune configuration file.
        figsize(tuple): The size of the figure on where the numerical channels are plotted.
            Default is (16,12).
        max_cols(int): Number of charts that should be plotted in one row. Default is 2.

    Attributes:
        context('neptune.Context'): Neptune context.
        config(`ParameterConfig): ParameterConfig object containt the parsed `parameters` section
            from the Neptune configuration file.
        numeric_channels(dict): Dictionary containing `channel_name` x,y values pairs where
            every y value was convertible to float.
        other_channels(dict): Dictionary containing `channel_name` x,y values pairs where
            not every y value was convertible to float.

    Examples:
        Create Neptune context instance:

        >>> import neptune
        >>> ctx = neptune.Context()

        Create Local Neptune context:

        >>> ctx = LocalNotebookContext(ctx, 'neptune.yaml')

        Access parameters from code:

        >>> ctx.params

        Send values to numerical channels and plot:

        >>> ctx.channel_send('loss', x=1, y=0.92)

    Note:
        After the development process is over you run your notebook as a
        python script and send the metrics to the Neptune app.
        Simply run the following command.

        $ jupyter nbconvert --to script main.py; neptune run --config neptune.yaml main.py
    """

    def __init__(self, context, config_filepath, figsize=(16, 12), max_cols=2):
        self.context = context
        self.config = ParameterConfig(config_filepath)
        self.numeric_channels = LocalNotebookChannels(figsize=figsize, max_cols=max_cols)
        self.other_channels = {}

    @property
    def params(self):
        """Parameters from config

        Examples:
            >>> import neptune
            >>> ctx = neptune.Context()
            >>> ctx = LocalNotebookContext(ctx, config_filepath='neptune.yaml')
            >>> ctx.params
            AttrDict({'lr': 0.2})
        """
        return self.config.params

    def channel_send(self, channel_name, x=None, y=None):
        """ Local notebook substitute for the in-app channel_send.

        Every value that is send via this method is automatically plotted in
        your notebook and added to the `numeric_channels` dictionary under the `channel_name` key.
        If the y value cannot be converted to `float` it will not be plotted but will be
        added to the `other_channels` dictionary under the `channel_name` key.

        Args:
            channel_name(str): The name of the channel
            x(int): The iteration or epoch. If None it will be infered from the y values.
            y(float or obj): The values that should be plotted. It is usually a loss or a metric.
                If the value passed to y cannot be converted to `float` it will be treated as `other_channel`
                and not displayed.

        Examples:
            Create Neptune context instance:

            >>> import neptune
            >>> ctx = neptune.Context()

            Create Local Neptune context:

            >>> ctx = LocalNotebookContext(ctx, 'neptune.yaml')

            Access parameters from code:

            >>> ctx.params

            Send values to numerical channels and plot:

            >>> ctx.channel_send('loss', x=1, y=0.92)

        Note:
            If `x=None` it will be infered from the y values.

        """
        try:
            y = float(y)
            self.numeric_channels[channel_name] = self.numeric_channels.get(channel_name,
                                                                            LocalNotebookChannel(channel_name))
            self.numeric_channels[channel_name].update({'x': x, 'y': y})
            self.numeric_channels.plot()
        except Exception:
            self.other_channels[channel_name] = self.other_channels.get(channel_name,
                                                                        LocalNotebookChannel(channel_name))
            self.other_channels[channel_name].update({'x': x, 'y': y})


class ParameterConfig(object):
    """Parsed Neptune configuration file.

    It is a simplified configuration file containg only the parsed parameters
    section.

    Args:
        filepath(str): Filepath to the Neptune configuration file.

    Attributes:
        filepath(str): Filepath to the Neptune configuration file.
        config(`attrdict.AttrDict`): Attribute dictionary containing the `parameters` from the
            configuration file.

    Examples:
        Create a config from `.yaml` file.

        >>> config = ParameterConfig('neptune.yaml')

        Access parameters from code:

        >>> config.params
        AttrDict({'lr': 0.2})

    """

    def __init__(self, filepath):
        assert os.path.exists(filepath), 'Specified Neptune configuration file {} does not exist'.format(filepath)

        self.filepath = filepath
        self.config = ParameterConfig.read_yaml(filepath)

        assert 'parameters' in self.config.keys(), 'Specified Neptune configuration file {} \
            is missing the parameters section'.format(filepath)

        self.params = self.config.parameters

    @staticmethod
    def read_yaml(filepath):
        with open(filepath) as f:
            config = yaml.load(f)
        assert config, 'Specified Neptune configuration file {} is empty'.format(filepath)
        return AttrDict(config)


class LocalNotebookChannels(dict):
    def __init__(self, figsize, max_cols):
        super().__init__(self)
        self.figsize = figsize
        self.max_cols = max_cols

    def plot(self):
        clear_output(wait=True)
        plt.figure(figsize=self.figsize)
        for metric_id, (channel_name, channel) in enumerate(self.items()):
            plt.subplot((len(self) + 1) // self.max_cols + 1, self.max_cols, metric_id + 1)
            plt.plot(channel.x, channel.y, label=channel_name)
            plt.legend()
        plt.show()


class LocalNotebookChannel(object):
    def __init__(self, channel_name):
        self.channel_name = channel_name
        self.y = []
        self.x = []

    def update(self, metrics):
        self.x.append(metrics['x'])
        self.y.append(metrics['y'])

        if None in self.x:
            self.x = list(range(len(self.y)))
            