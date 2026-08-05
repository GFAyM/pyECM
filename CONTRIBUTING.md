# Contributing to PyECM

## Development setup

Install pyECM in editable mode and install developers dependencies:

      ```pip install -e .```

      ```pip install -r requirements_dev.txt```

      ```pip install -r requirements_doc.txt```

## Running tests

Run all tests:

      ``tox``

Run short tests:

      ``tox -e fast``

## Linting and formatting

Check format files using flake8:

      ``tox -e flake8``

Format files according to docformatter, isort and black:

      ``tox -e format``


## Update the requirements lists

To update requirements.txt, requierements_dev.txt and requirements_doc.txt dependencies,

      ``pip-compile requirements.in``

      ``pip-compile requirements_dev.in``

      ``pip-compile requirements_doc.in``

## Building the docs locally

Either build the html files using doc:

      ``tox -e docs``

or executing under the doc folder:

      ``make html``

## Update the rst files

To update the rst files under the doc folder:

      ``sphinx-apidoc -f -o docs/source pyECM/``


