# Installation

## Requirements

- Python ≥ 3.13
- No system-level dependencies for the core library (excluding experimental SUMO components)

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

## Optional: SUMO (experimental components only)

The [experimental pipeline components](../experimental/index.md) require
SUMO to be installed separately. See the
[official SUMO installation guide](https://sumo.dlr.de/docs/Installing/index.html).
SUMO is **not** required for the core CTM/METANET simulation.
