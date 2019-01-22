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

import numpy as np
from PIL import Image


def fig2pil(fig):
    """Converts matplotlib fig to PIL.Image

    Args:
        fig(`matplotlib.pyplot.figure`): Any matplotlib figure.

    Returns:
        `PIL.Image`: figure, converted to PIL Image.

    Examples:
        Create a figure:

        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> import seaborn as sns
        >>> fig = plt.figure(figsize=(16,12))
        >>> sns.distplot(np.random.random(100))

        Convert to PIL.

        >>> pil_figure = fig2pil(fig)

    Note:
        On some machines, using this function has cause matplotlib errors.
        What helped every time was to change matplotlib backend by adding the following snippet
        towards the top of your script:

        >>> import matplotlib
        >>> matplotlib.use('Agg')
    """
    fig.canvas.draw()

    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)
    buf = np.roll(buf, 3, axis=2)

    w, h, _ = buf.shape
    return Image.frombytes("RGBA", (w, h), buf.tostring())


def is_offline_context(context):
    """Checks whether the context is offline.

    Args:
        context(`neptune.Context`): Neptune context.

    Returns:
        bool: Whether or not the context is offline.

    Examples:
        Create a Neptune context:

        >>> import neptune
        >>> context = neptune.Context()

        Check if it is offline:

        >>> is_offline_context(context)
        True
    """
    return context.params.__class__.__name__ == 'OfflineContextParams'
