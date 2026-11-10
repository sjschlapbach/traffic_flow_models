# Installation

## Requirements

- Python ≥ 3.13
- No system-level dependencies

## Install from PyPI

```bash
pip install traffic-flow-models
```

## Development install

```bash
git clone https://github.com/sjschlapbach/traffic_flow_models.git
cd traffic_flow_models
pip install -e ".[dev]"
```

The `dev` extras include `pytest`, `pylint`, `git-cliff`, and all documentation
dependencies (`mkdocs-material`, `mkdocstrings`, etc.).
