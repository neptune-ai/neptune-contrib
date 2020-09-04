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
    'log_video',
]


def log_video(path_to_file, video_name=None, experiment=None):
    """Logs a video file to 'artifacts/video' with player.

    Logs a video file to the 'artifacts/video' in the experiment, where you can play it directly from the browser.

    You can also download raw video file to the local machine.
    Just use "three vertical dots" located to the right from the player.

    Args:
        path_to_file (:obj:`str`): Path to video file.
        video_name (:obj:`str`, optional, default is ``None``): Name to be displayed in artifacts/video.
            | If `None`, file name is used.
        experiment (:obj:`neptune.experiments.Experiment`, optional, default is ``None``):
            | For advanced users only. Pass Neptune
              `Experiment <https://docs.neptune.ai/neptune-client/docs/experiment.html#neptune.experiments.Experiment>`_
              object if you want to control to which experiment data is logged.
            | If ``None``, log to currently active, and most recent experiment.

    Example:

        .. code:: python3

            log_video('video-file.mp4')
            log_video('/full/path/to/some/other/video/file.mp4')
            log_video('/full/path/to/some/other/video/file.mp4', 'my_video')

    Note:
        Check out how the logged video file looks in Neptune:
        `here <https://ui.neptune.ai/o/shared/org/showroom/e/
        SHOW-1542/artifacts?path=video%2F&file=jellyfish-25-mbps-hd-hevc.html>`_.

    Warning:
        Video files contribute to the storage usage. Be mindful with large video files.
    """

    import base64
    from io import StringIO

    _exp = experiment if experiment else neptune

    name, file_ext = os.path.split(path_to_file)[1].split('.')

    if video_name is None:
        video_name = name
    else:
        assert isinstance(video_name, str), 'video_name must be string, got {}'.format(type(video_name))

    encoded_video = base64.b64encode(open(path_to_file, 'rb').read())
    html = """<!DOCTYPE html>
        <html>
        <body>

        <video controls>
          <source src='data:video/{};base64,{}'>
        </video>

        </body>
        </html>""".format(file_ext, encoded_video.decode())

    buffer = StringIO(html)
    buffer.seek(0)

    _exp.log_artifact(buffer, 'video/{}.html'.format(video_name))
