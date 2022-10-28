clean:
	rm -fr .tox/ dist/ VERSION

build:
	python setup.py git_version sdist

tests: unit_tests

unit_tests:
	tox
