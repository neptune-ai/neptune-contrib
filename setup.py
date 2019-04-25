import os

from setuptools import find_packages, setup


def main():
    root_dir = os.path.dirname(__file__)

    with open(os.path.join(root_dir, 'requirements.txt')) as f:
        requirements = [r.strip() for r in f]
        setup(
            name='neptune-contrib',
            version='0.4.2',
            description='Neptune Python library contributions',
            author='neptune.ml',
            author_email='contact@neptune.ml',
            url="https://github.com/neptune-ml/neptune-contrib",
            long_description='Neptune Python library contributions',
            license='MIT License',
            install_requires=requirements,
            packages=find_packages(include=['neptunecontrib*']),
        )


if __name__ == "__main__":
    main()
