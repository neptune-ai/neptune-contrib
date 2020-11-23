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

import warnings

import neptune

__all__ = [
    'log_chart',
]


class PlotlyIncompatibilityException(Exception):
    pass


def log_chart(name, chart, experiment=None):
    """Logs charts from matplotlib, plotly, bokeh, and altair to neptune.

    Plotly, Bokeh, and Altair charts are converted to interactive HTML objects and then uploaded to Neptune
    as an artifact with path charts/{name}.html.

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

            neptune.init(api_token='ANONYMOUS',
                         project_qualified_name='shared/showroom')
            neptune.create_experiment(name='experiment_with_charts')

        Create matplotlib figure and log it to Neptune::

            import matplotlib.pyplot as plt

            fig = plt.figure()
            x = [21,22,23,4,5,6,77,8,9,10,31,32,33,34,35,36,37,18,49,50,100]
            plt.hist(x, bins=5)
            plt.show()

            from neptunecontrib.api import log_chart

            log_chart('matplotlib_figure', fig)

        Create Plotly chart and log it to Neptune::

            import plotly.express as px

            df = px.data.tips()
            fig = px.histogram(df, x="total_bill", y="tip", color="sex", marginal="rug",
                               hover_data=df.columns)
            fig.show()

            from neptunecontrib.api import log_chart

            log_chart('plotly_figure', fig)

        Create Altair chart and log it to Neptune::

            import altair as alt
            from vega_datasets import data

            source = data.cars()

            chart = alt.Chart(source).mark_circle(size=60).encode(
                            x='Horsepower',
                            y='Miles_per_Gallon',
                            color='Origin',
                            tooltip=['Name', 'Origin', 'Horsepower', 'Miles_per_Gallon']
            ).interactive()

            from neptunecontrib.api import log_chart

            log_chart('altair_chart', chart)

        Create Bokeh figure and log it to Neptune::

            from bokeh.plotting import figure

            p = figure(plot_width=400, plot_height=400)

            # add a circle renderer with a size, color, and alpha
            p.circle([1, 2, 3, 4, 5], [6, 7, 2, 4, 5], size=20, color="navy", alpha=0.5)

            from neptunecontrib.api import log_chart

            log_chart('bokeh_figure', p)

    Note:
        Check out how the logged charts look in Neptune:
        `example experiment
        <https://ui.neptune.ai/o/shared/org/showroom/e/SHOW-973/artifacts?path=charts%2F&file=bokeh_figure.html>`_
     """
    _exp = experiment if experiment else neptune

    if is_matplotlib_pyplot(chart) or is_matplotlib_figure(chart):
        if is_matplotlib_pyplot(chart):
            chart = chart.gcf()

        try:
            from plotly import tools

            # When Plotly cannot accurately convert a matplotlib plot, it emits a warning.
            # Then we want to fallback on logging the plot as an image.
            #
            # E.g. when trying to convert a Seaborn confusion matrix or a hist2d, it emits a UserWarning with message
            # "Dang! That path collection is out of this world. I totally don't know what to do with it yet!
            # Plotly can only import path collections linked to 'data' coordinates"
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "error",
                    category=UserWarning,
                    message=".*Plotly can only import path collections linked to 'data' coordinates.*")
                try:
                    chart = tools.mpl_to_plotly(chart)
                except AttributeError as e:
                    if "is_frame_like" in e.args[0]:
                        plotly_version = "unknown"
                        try:
                            import plotly
                            plotly_version = plotly.version.__version__
                        except:
                            pass
                        matplotlib_version = "unknown"
                        try:
                            import matplotlib
                            matplotlib_version = matplotlib.__version__
                        except:
                            pass
                        raise PlotlyIncompatibilityException(
                            "Unable to convert plotly figure to matplotlib format. "
                            "Your matplotlib ({}) and plotlib ({}) versions are not compatible. "
                            "See https://stackoverflow.com/q/63120058 for details."
                            .format(matplotlib_version, plotly_version))
                    else:
                        raise e

            _exp.log_artifact(export_plotly_figure(chart), "charts/" + name + '.html')
        except ImportError:
            print("Plotly not installed. Logging plot as an image.")
            _exp.log_artifact(export_matplotlib_figure(chart), "charts/" + name + '.png')
        except UserWarning:
            print("Couldn't convert Matplotlib plot to interactive Plotly plot. Logging plot as an image instead.")
            _exp.log_artifact(export_matplotlib_figure(chart), "charts/" + name + '.png')

    elif is_plotly_figure(chart):
        _exp.log_artifact(export_plotly_figure(chart), "charts/" + name + '.html')

    elif is_bokeh_figure(chart):
        _exp.log_artifact(export_bokeh_figure(chart), "charts/" + name + '.html')

    elif is_altair_chart(chart):
        _exp.log_artifact(export_altair_chart(chart), "charts/" + name + '.html')

    else:
        raise ValueError("Currently supported are matplotlib, plotly, altair, and bokeh figures")


def is_matplotlib_pyplot(chart):
    return hasattr(chart, '__name__') and chart.__name__.startswith('matplotlib.')


def is_matplotlib_figure(chart):
    return chart.__class__.__module__.startswith('matplotlib.') and chart.__class__.__name__ == 'Figure'


def is_plotly_figure(chart):
    return chart.__class__.__module__.startswith('plotly.') and chart.__class__.__name__ == 'Figure'


def is_altair_chart(chart):
    return chart.__class__.__module__.startswith('altair.') and 'Chart' in chart.__class__.__name__


def is_bokeh_figure(chart):
    return chart.__class__.__module__.startswith('bokeh.') and chart.__class__.__name__ == 'Figure'


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


def export_altair_chart(chart):
    from io import StringIO

    buffer = StringIO()
    chart.save(buffer, format='html')
    buffer.seek(0)

    return buffer


def export_bokeh_figure(chart):
    from io import StringIO
    from bokeh.resources import CDN
    from bokeh.embed import file_html

    html = file_html(chart, CDN)
    buffer = StringIO(html)
    buffer.seek(0)

    return buffer
