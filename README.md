# Macroscopic Traffic Flow Models

<!-- TODO: Add brief description of the repository -->

## Setup

Consider setting up a virtual environment for this project. The project was developed with Python 3.13 and pip 25.3. The following commands will create a virtual environment and install the required dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -e .
```

When using an operating system that has Python pre-installed (e.g. macOS), you might have to manually install the corresponding python version and create a virtual environment with that version.

```bash
brew install python@3.13
python3.13 -m venv venv
source venv/bin/activate
pip install -e .
```

In some cases, it might be necessary to set the `PYTHONPATH` environment variable to the root of this repository. This can be done by running the following command:

```bash
export PYTHONPATH=.
```

## Test Suite

After installing the pytest package, the test suite can be run with the following command:

```bash
pytest src/test/
```
