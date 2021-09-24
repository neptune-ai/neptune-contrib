from setuptools import find_packages, setup
import versioneer


def main():
    with open('README.md') as readme_file:
        readme = readme_file.read()

    extras = {
        'monitoring': ['pytorch-lightning>=1.0.0'],
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
