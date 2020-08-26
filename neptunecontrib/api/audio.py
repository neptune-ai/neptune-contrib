#
# Copyright (c) 2020, Neptune Labs Sp. z o.o.
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

import neptune

__all__ = [
    'log_audio',
]


def log_audio(path_to_file, audio_name=None, experiment=None):
    """Logs audio file to 'artifacts/audio' with player.

    Logs audio file to the 'artifacts/audio' in the experiment, where you can play it directly from the browser.
    You can also download raw audio file to the local machine.
    Just use "three vertical dots" located to the right from the player.

    Args:
        path_to_file (:obj:`str`): Path to audio file.
        audio_name (:obj:`str`, optional, default is ``None``): Name to be displayed in artifacts/audio.
            | If `None`, file name is used.
        experiment (:obj:`neptune.experiments.Experiment`, optional, default is ``None``):
            | For advanced users only. Pass Neptune
              `Experiment <https://docs.neptune.ai/neptune-client/docs/experiment.html#neptune.experiments.Experiment>`_
              object if you want to control to which experiment data is logged.
            | If ``None``, log to currently active, and most recent experiment.

    Example:

        .. code:: python3

            log_audio('audio-file.wav')
            log_audio('/full/path/to/some/other/audio/file.mp3')
            log_audio('/full/path/to/some/other/audio/file.mp3', 'my_audio')

    Note:
        Check out how the logged audio file looks in Neptune:
        `here <https://ui.neptune.ai/o/shared/org/showroom/e/SHOW-1485/artifacts?path=audio%2F>`_.
    """

    import base64
    from io import StringIO

    _exp = experiment if experiment else neptune

    name, file_ext = os.path.split(path_to_file)[1].split('.')

    if audio_name is None:
        audio_name = name
    else:
        assert isinstance(audio_name, str), 'audio_name must be string, got {}'.format(type(audio_name))

    encoded_sound = base64.b64encode(open(path_to_file, 'rb').read())
    html = """<!DOCTYPE html>
        <html>
        <body>

        <audio controls>
          <source src='data:audio/{};base64,{}'>
        </audio>

        </body>
        </html>""".format(file_ext, encoded_sound.decode())

    buffer = StringIO(html)
    buffer.seek(0)

    _exp.log_artifact(buffer, 'audio/{}.html'.format(audio_name))
