## [1.0.0] - 2026-04-27

### Enhancements

- *(network)* Refactor network structure to be based on a linked list for improved generality (#17)
- *(network)* Add vkt and vht computation logic compatible with new network structure (#22)
- *(metanet)* Introduce symbolic parametrization for model update function (#23)
- *(metanet)* Add term to represent speed reductions caused by merging effects following onramps (#25)
- *(ctm)* Introduce flow boundary conditions at destination links to model ctm supply limits (#39)
- *(network)* Improve network structure visualization to take into account node positions (#42)
- *(network)* Enable storing and loading network structures from json file (#43)
- Add possibility to visualize multiple simulation scenarios next to each other (#46)
- *(demand-aggregator)* Compute macroscopic model demand based on sliding window aggregation of microscopic simulation results (#53)
- Introduce node between origin and onramp for unified demand interfaces (#54)
- Create more uniform network structure with node between offramp and destination (#55)
- Set up integration of microsimulation-based ground truth run with METANET calibrator (#61)
- Add tracking of upstream and downstream onramps for coordinated ramp metering (#66)
- Improve visualizations of simulation results by including node inflows for link overview (#71)
- Extend demand model with the possibility to manually specify highway-to-highway demand (#69)
- Extend demand model with the possibility to manually specify highway-to-highway demand (#73)
- Modify pipeline to support direct specification of sumo network and routing files (#76)
- Add side-by-side visualization of micro and macro simulation results (#70)
- Explicitly compute highway-only VHT and VKT values (#84)
- *(calibrator)* Introduce visualization of parameter calibration convergence and generalize calibrator for improved model independence (#85)
- *(pipeline)* Add possibility to include parameter search in pipeline execution

### Features

- *(simulation)* Add classes for SUMO scenario generation and simulation for given city (#16)
- Introduce nodes and links to network structure for METANET compatibility (#21)
- *(ctm)* Migrate CTM model to be compatible with new node- and link-based network structure (#32)
- Introduce network arbitrator, demand aggregator and loop detector generator for network extraction from SUMO and demand aggregation from SUMO outputs
- Introduce animated illusttration of macroscropic flow model simulation (#44)
- Implement model calibration functionality for METANET-based scenarios based on simulation results (#45)
- Add sumo microsimulation-based computation of turning rates at nodes (#49)
- Add demand aggregation for macroscopic model calibration based on microscopic simulation results (#57)
- Implement basic flow rate controller and ALINEA compatible with new structures (#63)
- Add possibility to define arbitrary custom controller for single onramp (#64)
- Implement METALINE coordinated ramp metering algorithm (#75)
- Compute flow and density boundary conditions at destinations (#65)

### Bug Fixes

- *(metanet)* Resolve issues with the node upstream speed computation and other minor fixes
- *(metanet)* Ensure that correct node is chosen for computation of virtual density computations downstream of motorway link (#26)
- *(metanet)* Ensure that origin network inflows are limited through first outgoing link parameters (#28)
- *(metanet)* Ensure that nodes with onramps are only connected to a single outgoing motorway link (#29)
- *(metanet)* Make sure that singular denominators are handled correctly in node flow computations (#30)
- *(metanet)* Resolve issues in node upstream speed update if no upstream flow was measured (#31)
- *(ctm)* Ensure that node downstream supply restrictions are correctly computed for outgoing motorway links (#34)
- *(ctm/metanet)* Ensure that causality assumptions regarding off-ramps and destinations are fulfilled (#38)
- *(network)* Ensure that typing of interfaces for network state initialization is correct
- Ensure that short links from SUMO network are either stretched or eliminated (#50)
- Ensure that detector data for demand aggregation is identified correctly (#51)
- *(demo)* Ensure that detector output file paths are consistent
- Ensure that onramps and origins are handled correctly for demand aggregation (#56)
- Ensure that shortest path search in demand aggregator is efficient to remain tractable (#59)
- Resolve issue that onramps and offramps were not visible in video visualizations of larger scenarios
- Update log structure to be compatible with simulation output log structure (#60)
- Ensure that units of ALINEA are consistent for flow restriction formulation (#74)
- Modified origin demand agrgegation and simplified highway and urban demand generation (#78)
- Resolve merged edge error and missing origin demand function (#79)
- *(ctm)* Resolve issue with store-and-forward update of off-ramp queues (#80)
- Resolve double counting of demand by the inflow detectors
- *(metanet)* Ensure that nodes are processed in correct order to fulfill assumptions (#82)

### Refactor

- *(motorwaylink)* Change naming of capacity-related quantities for consistency (#33)
- Extract simulation functionalities and corresponding visualizations to dedicated class (#47)
- Change physical network specifications definition to file-based approach (#48)
- Update structure of scenarios for increased flexibility for complex setups (#67)

### Miscellaneous Tasks

- *(release)* Update changelog for version 0.1.3
- Update README with SUMO installation requirements for pipeline
- *(ci)* Enable demo script execution in CI pipeline (#19)
- Introduce python linting and improve naming of motorway link components (#20)
- *(tests)* Add extensive testing coverage for new network structure (#24)
- *(network)* Add functionality to log numerical simulation results
- *(network)* Add validation that nodes connected to origins have exactly one outgoing motorway link
- *(ci)* Simplify linting action setup (#41)
- Update README with up-to-date examples of library usage
- Remove IDE configuration parameters from repository
- Relicense to MIT
- Update README with examples for new network structure
- Refine virtual downstream density computation for store-and-forward links in METANET (#62)
- Add scenario choice through CLI arguments
- Extend metanet demonstration script to support all scenarios
- Update README with new pipeline CI action status
- Add license to python project file
- Revert changes to demand model
- Update license mention in README
- Update README with new ramp metering controller interface structure
- Do not plot node summary and inflows for origin-onramp nodes
- Ensure that all calibration figures are generated for pipeline run with METANET model
- Update simulation video plotting range to avoid distortion and timer cutoff
- Extend the pipeline scenarios with METANET model variants (#83)
- Store link and node plots in dedicated results subdirectories for better overview
- Add section on calibration to README
- [**breaking**] Update versioning logic
- *(release)* V1.0.0
## [0.1.3] - 2025-12-28

### Miscellaneous Tasks

- *(release)* Update changelog for version 0.1.2
- Update README and release workflow for correct creation of release notes (#14)
- Update README with package-based installation instructions
- Update gitignore
- *(release)* V0.1.3
## [0.1.2] - 2025-12-17

### Enhancements

- Add functionality to plot a network structure including ramps (#2)
- Support offramps and split ratios in cell transmission model (#4)
- Add demo scenarios for macroscopic flow simulation (#6)
- Extend network flow plots with onramp and offramp flows
- *(ctm)* Update CTM model equations to ensure causality during cell updates (#10)
- *(metanet)* Update the model equations to maintain causality
- *(network)* Ensure that CFL condition is satisfied for all each network

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
