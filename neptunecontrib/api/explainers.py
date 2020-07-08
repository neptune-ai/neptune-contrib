import neptune

from neptunecontrib.api import log_chart

__all__ = [
    'log_explainer',
    'log_local_explanations',
    'log_global_explanations'
]


def log_explainer(filename, explainer, experiment=None):
    """Logs dalex explainer to Neptune.

    Dalex explainer is pickled and logged to Neptune.

    Args:
        filename (:obj:`str`): filename that will be used as an artifact's destination.
        explainer (:obj:`dalex.Explainer`): and instance of dalex explainer
        experiment (:obj:`neptune.experiments.Experiment`, optional, default is ``None``):
            | For advanced users only. Pass Neptune
              `Experiment <https://docs.neptune.ai/neptune-client/docs/experiment.html#neptune.experiments.Experiment>`_
              object if you want to control to which experiment data is logged.
            | If ``None``, log to currently active, and most recent experiment.

    Examples:
        Start an experiment::

            import neptune

            neptune.init(api_token='ANONYMOUS',
                         project_qualified_name='shared/dalex-integration')
            neptune.create_experiment(name='logging explanations')

        Train your model and create dalex explainer::

            ...
            clf.fit(X, y)

            exp = dx.Explainer(clf, X, y, label="Titanic MLP Pipeline")

            log_local_explanations(exp, new_observation)
            log_global_explanations(exp)
            log_explainer('explainer.pkl', exp)

    Note:
        Check out how the logged explainer looks in Neptune:
        `example experiment <https://ui.neptune.ai/o/shared/org/dalex-integration/e/DAL-48/artifacts>`_
     """
    _exp = experiment if experiment else neptune

    neptune.log_artifact(export_dalex_explainer(explainer), filename)


def export_dalex_explainer(explainer):
    from io import BytesIO

    buffer = BytesIO()
    explainer.dump(buffer)
    buffer.seek(0)

    return buffer


def log_local_explanations(explainer, observation, experiment=None):
    """Logs local explanations from dalex to Neptune.

    Dalex explanations are converted to interactive HTML objects and then uploaded to Neptune
    as an artifact with path charts/{name}.html.

    The following explanations are logged: break down, break down with interactions, shap, ceteris paribus, and ceteris paribus for categorical variables.
    Explanation charts are created and logged with default settings. To log charts with custom settings, create a custom chart and use `neptunecontrib.api.log_chart`.
    For more information about Dalex go to `Dalex Website <https://modeloriented.github.io/DALEX/>`_.

    Args:
        explainer (:obj:`dalex.Explainer`): and instance of dalex explainer
        observation (:obj): and observation that can be fed to the classifier or regressor for which the explainer was created
        experiment (:obj:`neptune.experiments.Experiment`, optional, default is ``None``):
            | For advanced users only. Pass Neptune
              `Experiment <https://docs.neptune.ai/neptune-client/docs/experiment.html#neptune.experiments.Experiment>`_
              object if you want to control to which experiment data is logged.
            | If ``None``, log to currently active, and most recent experiment.

    Examples:
        Start an experiment::

            import neptune

            neptune.init(api_token='ANONYMOUS',
                         project_qualified_name='shared/dalex-integration')
            neptune.create_experiment(name='logging explanations')

        Train your model and create dalex explainer::

            ...
            clf.fit(X, y)

            exp = dx.Explainer(clf, X, y, label="Titanic MLP Pipeline")

            new_observation = pd.DataFrame({'gender': ['male'],
                                            'age': [25],
                                            'class': ['1st'],
                                            'embarked': ['Southampton'],
                                            'fare': [72],
                                            'sibsp': [0],
                                            'parch': 0},
                                           index=['John'])

            log_local_explanations(exp, new_observation)

    Note:
        Check out how the logged explanations look in Neptune:
        `example experiment <https://ui.neptune.ai/o/shared/org/dalex-integration/e/DAL-48/artifacts?path=charts%2F>`_
     """
    _exp = experiment if experiment else neptune

    bd = explainer.predict_parts(observation, type='break_down')
    bd_interactions = explainer.predict_parts(observation, type='break_down_interactions')
    sh = explainer.predict_parts(observation, type='shap')
    cp = explainer.predict_profile(observation)

    bd_plot = bd.plot(show=False)
    log_chart(name='Break Down', chart=bd_plot, experiment=_exp)

    bd_interactions_plot = bd_interactions.plot(show=False)
    log_chart(name='Break Down Interactions', chart=bd_interactions_plot, experiment=_exp)

    sh_plot = sh.plot(show=False)
    log_chart(name='SHAP', chart=sh_plot, experiment=_exp)

    cp_plot = cp.plot(show=False)
    log_chart(name="Ceteris Paribus", chart=cp_plot, experiment=_exp)

    cp_plot_cat = cp.plot(variable_type="categorical", show=False)
    log_chart(name="Ceteris Paribus Categorical", chart=cp_plot_cat, experiment=_exp)


def log_global_explanations(explainer, categorical_features=None, experiment=None):
    """Logs global explanations from dalex to Neptune.

    Dalex explanations are converted to interactive HTML objects and then uploaded to Neptune
    as an artifact with path charts/{name}.html.

    The following explanations are logged: variable importance. If categorical features are specified partial dependence and accumulated dependence are also logged.
    Explanation charts are created and logged with default settings. To log charts with custom settings, create a custom chart and use `neptunecontrib.api.log_chart`.
    For more information about Dalex go to `Dalex Website <https://modeloriented.github.io/DALEX/>`_.

    Args:
        explainer (:obj:`dalex.Explainer`): and instance of dalex explainer
        categorical_features (:list): list of categorical features for which you want to create partial and accumulated dependence plots.
        experiment (:obj:`neptune.experiments.Experiment`, optional, default is ``None``):
            | For advanced users only. Pass Neptune
              `Experiment <https://docs.neptune.ai/neptune-client/docs/experiment.html#neptune.experiments.Experiment>`_
              object if you want to control to which experiment data is logged.
            | If ``None``, log to currently active, and most recent experiment.

    Examples:
        Start an experiment::

            import neptune

            neptune.init(api_token='ANONYMOUS',
                         project_qualified_name='shared/dalex-integration')
            neptune.create_experiment(name='logging explanations')

        Train your model and create dalex explainer::

            ...
            clf.fit(X, y)

            exp = dx.Explainer(clf, X, y, label="Titanic MLP Pipeline")
            log_global_explanations(exp, categorical_features=["gender", "class"])

    Note:
        Check out how the logged explanations look in Neptune:
        `example experiment <https://ui.neptune.ai/o/shared/org/dalex-integration/e/DAL-48/artifacts?path=charts%2F>`_
     """
    _exp = experiment if experiment else neptune

    vi = explainer.model_parts()

    vi_plot = vi.plot(show=False)
    log_chart(name="Variable Importance", chart=vi_plot, experiment=_exp)

    if categorical_features:
        pdp_num = explainer.model_profile(type='partial')
        pdp_num.result["_label_"] = 'pdp'

        pdp_cat = explainer.model_profile(type='partial', variable_type='categorical', variables=categorical_features)
        pdp_cat.result['_label_'] = 'pdp'

        ale_cat = explainer.model_profile(type='accumulated', variable_type='categorical',
                                          variables=categorical_features)
        ale_cat.result['_label_'] = 'ale'

        ale_num = explainer.model_profile(type='accumulated')
        ale_num.result["_label_"] = 'ale'

        pdp_num_plot = pdp_num.plot(ale_num, show=False)
        log_chart(name="Partial Dependence", chart=pdp_num_plot, experiment=_exp)

        ale_cat_plot = ale_cat.plot(pdp_cat, show=False)
        log_chart(name="Accumulated Dependence", chart=ale_cat_plot, experiment=_exp)
