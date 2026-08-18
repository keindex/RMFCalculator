---
title: "RMFCalculator"
---
# RMFCalculator

Relativistic Mean Field Calculator for Neutron Star Equation of State.

## Installation

```bash
pip install .
```

## Quick Start

```python
import numpy as np
from RMFCalculator.eos import compute_EOS, PolyInterpolate
from RMFCalculator.tov import OutputMR

# FSU parameters
FSU_params = {
    'm_sigma': 491.5 / 197.327,
    'm_omega': 782.5 / 197.327,
    'm_rho': 763.0 / 197.327,
    'g_sigma': 10.217,
    'g_omega': 13.584,
    'g_rho': 8.919,
    'kappa': 1.950e-4,
    'lambda_0': 5.651e-3,
    'zeta': 3.483e-3,
    'Lambda_w': 1.864e-2,
}

theta = np.array([
    FSU_params['m_sigma'], FSU_params['m_omega'], FSU_params['m_rho'],
    FSU_params['g_sigma'], FSU_params['g_omega'], FSU_params['g_rho'],
    FSU_params['kappa'], FSU_params['lambda_0'], FSU_params['zeta'], FSU_params['Lambda_w']
])

# Compute EOS
eps, pres = compute_EOS(eps_crust, pres_crust, theta)

# Compute Mass-Radius relation
MR = OutputMR("", eps, pres)
```

## Features

- RMF (Relativistic Mean Field) EOS generation
- TOV (Tolman-Oppenheimer-Volkoff) solver for neutron star structure
- Multiple RMF parameter sets (FSU, FSU2, FSU2R, FSU Gold)
- Crust EOS interpolation with polytrope
- Mass-Radius and tidal deformability calculations
