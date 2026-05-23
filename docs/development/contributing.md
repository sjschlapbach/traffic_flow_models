# Contributing

## Development install

```bash
git clone https://github.com/sjschlapbach/traffic_flow_models.git
cd traffic_flow_models
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

## Running tests

```bash
source venv/bin/activate
pytest
```

Run a specific test file:

```bash
pytest tests/test_ctm.py -v
```

## Code style

The project uses [Black](https://black.readthedocs.io/) for linting and formatting:

```bash
black .
```

## Adding documentation

Documentation lives in `docs/`. Serve locally with:

```bash
mkdocs serve
```

## Pull requests

1. Fork the repository and create a feature branch.
2. Ensure all tests pass and new code is covered.
3. Open a pull request against `dev`. The CI pipeline will build the documentation
   and run the test suite.
