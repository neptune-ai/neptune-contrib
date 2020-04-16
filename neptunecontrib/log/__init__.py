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


def log_chart(name, chart):
    if is_matplotlib_pyplot(chart) or is_matplotlib_figure(chart):
        if is_matplotlib_pyplot(chart):
            chart = chart.gcf()

        try:
            from plotly import tools
            chart = tools.mpl_to_plotly(chart)

            neptune.log_artifact(export_plotly_figure(chart), "charts/" + name + '.html')
        except ImportError:
            neptune.log_artifact(export_matplotlib_figure(chart), "charts/" + name + '.png')

    elif is_plotly_figure(chart):
        neptune.log_artifact(export_plotly_figure(chart), "charts/" + name + '.html')

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
