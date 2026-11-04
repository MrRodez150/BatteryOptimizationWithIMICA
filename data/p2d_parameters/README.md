# P2D reference OCP tables

The two CSV files in this directory are copied from the PyBaMM repository at
commit `b4b6089a65f736fe2444ecf4536a297cbfbd3420`:

- `lico2_ocp_Ai2020.csv`: LCO open-circuit voltage data identified in its source
  header as Rieger et al. (2016).
- `graphite_ocp_Enertech_Ai2020.csv`: graphite OCP data distributed with the
  Enertech Ai2020 parameter set; its source file marks the experimental points.

The source parameter set cites Ai et al. (2020), *Journal of The
Electrochemical Society*, 167(1), 013512,
DOI [10.1149/2.0122001JES](https://doi.org/10.1149/2.0122001JES), and Rieger et
al. (2016), *Journal of The Electrochemical Society*, 163(8), A1566–A1575.
Ai2020 itself cautions that its parameter values are fitted to reported
experiments and are not asserted to represent true parameter values.

The LCO OCP is loaded only over the overlap `0.43 <= theta <= 0.9936`, matching
the stated domain of the Ai2020 LCO entropy fit. The graphite OCP is used only
isothermally at 298.15 K because a source-verified graphite entropy curve has
not yet been added. Neither table establishes validity for the user's target
cell. They are a reproducible reference parameterization for code and
verification work, not an external validation result.

PyBaMM's package license is BSD-3-Clause; its license is reproduced in
`LICENSE.txt`. The OCP tables retain their original source headers.
