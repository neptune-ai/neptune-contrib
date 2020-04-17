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

import neptune


def log_chart(name, chart, experiment=None):
    """Logs charts from matplotlib, plotly to neptune.

    Plotly figures are converted to interactive HTML and then uploaded to Neptune as an artifact with path
    charts/{name}.html.

    Matplotlib figures are converted optionally. If plotly is installed, matplotlib figures are converted
    to plotly figures and then converted to interactive HTML and uploaded to Neptune as an artifact with
    path charts/{name}.html. If plotly is not installed, matplotlib figures are converted to PNG images
    and uploaded to Neptune as an artifact with path charts/{name}.png

    Args:
        name (:obj:`str`):
            | Name of the chart (without extension) that will be used as a part of artifact's destination.
        chart (:obj:`matplotlib` or :obj:`plotly` Figure):
            | Figure from `matplotlib` or `plotly`. If you want to use global figure from `matplotlib`, you
              can also pass reference to `matplotlib.pyplot` module.
        experiment (:obj:`neptune.experiments.Experiment`, optional, default is ``None``):
            | For advanced users only. Pass Neptune
              `Experiment <https://docs.neptune.ai/neptune-client/docs/experiment.html#neptune.experiments.Experiment>`_
              object if you want to control to which experiment data is logged.
            | If ``None``, log to currently active, and most recent experiment.

    Examples:
        Start an experiment::

            import neptune

            neptune.init(project_qualified_name='USER_NAME/PROJECT_NAME')
            neptune.create_experiment(name='experiment_with_chart')

        Create some figure::

            import matplotlib.pyplot as plt

            plt.plot([1, 2, 3, 4])
            plt.ylabel('some numbers')

        Log the figure to Neptune::

            from neptunecontrib.logging.chart import log_chart

            log_chart('matplotlib_chart', plt)
     """
    _exp = experiment if experiment else neptune

    if is_matplotlib_pyplot(chart) or is_matplotlib_figure(chart):
        if is_matplotlib_pyplot(chart):
            chart = chart.gcf()

        try:
            from plotly import tools
            chart = tools.mpl_to_plotly(chart)

            _exp.log_artifact(export_plotly_figure(chart), "charts/" + name + '.html')
        except ImportError:
            _exp.log_artifact(export_matplotlib_figure(chart), "charts/" + name + '.png')

    elif is_plotly_figure(chart):
        _exp.log_artifact(export_plotly_figure(chart), "charts/" + name + '.html')

    else:
        raise ValueError("Currently supported are matplotlib and plotly figures")


def is_matplotlib_pyplot(chart):
    return hasattr(chart, '__name__') and chart.__name__.startswith('matplotlib.')


def is_matplotlib_figure(chart):
    return chart.__class__.__module__.startswith('matplotlib.') and chart.__class__.__name__ == 'Figure'


def is_plotly_figure(chart):
    return chart.__class__.__module__.startswith('plotly.') and chart.__class__.__name__ == 'Figure'


def export_plotly_figure(chart):
    from io import StringIO

    buffer = StringIO()
    chart.write_html(buffer)
    buffer.seek(0)

    return buffer


def export_matplotlib_figure(chart):
    from io import BytesIO

    buffer = BytesIO()
    chart.savefig(buffer, format='png')
    buffer.seek(0)

    return buffer
