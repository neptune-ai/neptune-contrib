from setuptools import find_packages, setup
import versioneer


def main():
    with open('README.md') as readme_file:
        readme = readme_file.read()

    extras = {
        'bots': ['python-telegram-bot'],
        'hpo': ['scikit-optimize>=0.5.2', 'scipy'],
        'monitoring': ['scikit-optimize>=0.7.4', 'sacred>=0.7.5', 'scikit-learn>=0.21.3', 'keras-tuner>=1.0.2',
                       'scikit-plot>=0.3.7', 'seaborn>=0.8.1', 'aif360>=0.2.1', 'xgboost>=0.82',
                       'yellowbrick>=1.2'],
        'versioning': ['boto3', 'numpy'],
        'viz': ['altair>=2.3.0', 'hiplot>=0.1.5'],
    }

    all_deps = []
    for group_name in extras:
        all_deps += extras[group_name]
    extras['all'] = all_deps

    base_libs = ['attrdict>=2.0.0', 'neptune-client>=0.4.126', 'joblib>=0.13', 'pandas', 'matplotlib',
                 'Pillow>=6.2.0']

    setup(
        name='neptune-contrib',
        version=versioneer.get_version(),
        cmdclass=versioneer.get_cmdclass(),
        description='Neptune.ai contributions library',
        author='neptune.ai',
        support='contact@neptune.ai',
        author_email='contact@neptune.ai',
        url="https://github.com/neptune-ai/neptune-contrib",
        long_description=readme,
        long_description_content_type="text/markdown",
        license='MIT License',
        install_requires=base_libs,
        extras_require=extras,
        packages=find_packages(include=['neptunecontrib*']),
    )


if __name__ == "__main__":
    main()
