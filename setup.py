from setuptools import find_packages, setup


def main():
    with open('README.md') as readme_file:
        readme = readme_file.read()

    extras = {
        'bots': ['python-telegram-bot'],
        'hpo': ['scikit-optimize==0.5.2', 'scipy'],
        'monitoring': ['scikit-optimize==0.5.2', 'sacred==0.7.5', 'scikit-learn==0.21.3',
                       'scikit-plot==0.3.7', 'seaborn==0.8.1', 'aif360==0.2.1'],
        'versioning': ['boto3', 'numpy'],
        'viz': ['altair==2.3.0'],
    }

    all_deps = []
    for group_name in extras:
        all_deps += extras[group_name]
    extras['all'] = all_deps

    base_libs = ['attrdict==2.0.0', 'neptune-client', 'joblib==0.13', 'pandas', 'matplotlib', 'Pillow==5.4.1']

    setup(
        name='neptune-contrib',
        version='0.14.0',
        description='Neptune.ml contributions library',
        author='neptune.ml',
        support='contact@neptune.ml',
        author_email='contact@neptune.ml',
        url="https://github.com/neptune-ml/neptune-contrib",
        long_description=readme,
        license='MIT License',
        install_requires=base_libs,
        extras_require=extras,
        packages=find_packages(include=['neptunecontrib*']),
    )


if __name__ == "__main__":
    main()
