# Contributing to PyECM

## Development setup

Install pyECM in editable mode and install developers dependencies.

1. Install pyECM in editable mode:

   ```bash
   pip install -e .
   ```

2. Install developer dependencies:

   ```bash
   pip install -r requirements_dev.txt
   ```

3. Install documentation dependencies:

   ```bash
   pip install -r requirements_doc.txt
   ```

## Running tests

Run all tests:

   ```bash
   tox
   ```

Run short tests:

   ```bash
   tox -e fast
   ```

## Linting and formatting

Check format files using flake8:

   ```bash
   tox -e flake8
   ```

Format files according to docformatter, isort and black:

   ```bash
   tox -e format
   ```


## Update the requirements lists

To update requirements.txt, requierements_dev.txt and requirements_doc.txt dependencies.

1. Update requirements.txt:

   ```bash
   pip-compile requirements.in
   ```

2. Update requirements_dev.txt:

   ```bash
   pip-compile requirements_dev.in
   ```

3. Update requirements_doc.txt:

   ```bash
   pip-compile requirements_doc.in
   ```
      

## Building the docs locally

Either build the html files using doc:

   ```bash
   tox -e docs
   ```

or executing under the doc folder:

   ```bash
   make html
   ```

## Update the rst files

To update the rst files under the doc folder:

   ```bash
   sphinx-apidoc -f -o docs/source pyECM/
   ```


