## [0.1.2] - 2025-12-17

### Features

- Develop first version of network model for traffic flow model application
- Add cell transmission model (#3)
- Add possibility to simulate the flows in a network using a macroscopic flow model (#5)
- Introduce METANET flow model (#7)

### Bug Fixes

- Ensure that alinea controller can be defined specific to an onramp
- Ensure that networks with offramps are handled correctly in METANET and CTM models (#9)
- *(ctm)* Use the backward wave speed of the downstream cell to compute the supply of space for flow update
- *(test)* Update interfaces in ctm tests to be consistent with class definitions
- *(ci)* Ensure that version check of release workflow is consistent with changelog formatting

### Other

- Add functionality to plot a network structure including ramps (#2)
- Support offramps and split ratios in cell transmission model (#4)
- Add demo scenarios for macroscopic flow simulation (#6)
- Extend network flow plots with onramp and offramp flows
- *(ctm)* Update CTM model equations to ensure causality during cell updates (#10)
- *(metanet)* Update the model equations to maintain causality
- *(network)* Ensure that CFL condition is satisfied for all each network

### Refactor

- Simplify computation of regulated onramp flows

### Miscellaneous Tasks

- Add github actions for python testing and format checks (#1)
- Add docstrings to all network component classes
- Add missing docstrings to network class methods
- Update README with more extensive description of repository (#8)
- Add input validation for object instantiation of network components
- Add scenario with on- and offramps
- Add license
- Add performance metrics and corresponding visualizations
- Update license mention in README
- *(ctm)* Assume free flow conditions for cells without vehicles
- Remove unused ctm test case
- Update scenarios to use critical density if no static density setpoint is defined for ALINEA control (#11)
- *(ci)* Setup automated release pipeline (#12)
- Update readme with build status badge
- Add status badge with license to readme
- *(ci)* Update release workflow with specification of branch to push changelog updates to
- *(release)* Bump pip package version for release
- *(release)* Bump pip package version for release
