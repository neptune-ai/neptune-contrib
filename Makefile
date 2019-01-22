clean:
	rm -fr .tox/ dist/ VERSION

build:
	python setup.py git_version sdist

tests: checkstyle_tests unit_tests

checkstyle_tests:
	python -m pylint -j 0 -f parseable neptunecontrib tests

unit_tests:
	tox
