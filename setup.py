from setuptools import find_packages, setup


def main():
    extras = {
        'bots': ['python-telegram-bot'],
        'hpo': ['scikit-optimize==0.5.2', 'scipy'],
        'monitoring': ['scikit-optimize==0.5.2', 'sacred==0.7.5', 'scikit-plot==0.3.7', 'seaborn'],
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
        version='0.11.0',
        description='Neptune Python library contributions',
        author='neptune.ml',
        author_email='contact@neptune.ml',
        url="https://github.com/neptune-ml/neptune-contrib",
        long_description='Neptune Python library contributions',
        license='MIT License',
        install_requires=base_libs,
        extras_require=extras,
        packages=find_packages(include=['neptunecontrib*']),
    )


if __name__ == "__main__":
    main()
