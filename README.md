# Tight-Binding Bethe-Salpeter Equation (TB-BSE) Solver

**TB-BSE** is a Python-based computational tool designed to solve the Bethe-Salpeter Equation for two-dimensional (2D) materials and vertical heterostructures (such as GaAs/MX₂). It calculates excitonic wavefunctions, binding energies, and optical absorption spectra using tight-binding parameters extracted from Density Functional Theory (DFT) and Wannier90.

## Key Features

- **Rigorous Wannier90 Integration**: 
  - Reads tight-binding Hamiltonians directly from `wannier90_hr.dat`.
  - Uses exact Wigner-Seitz supercell vectors (`wannier90_wsvec.dat`) to rigorously preserve point-group symmetries.
  - Implements sub-cell Coulomb gauge corrections using exact Wannier centres (`wannier90.wout`).
- **Macroscopic Dielectric Screening**: Employs the Rytova-Keldysh potential tailored for 2D films and heterostructures, accounting for the surrounding dielectric environment (e.g., substrate/vacuum).
- **Accurate Optics**: Calculates oscillator strengths using transition velocity matrix elements derived analytically via the Peierls substitution. Supports polarization-dependent absorption calculations (in-plane $x, y$ or out-of-plane $z$).
- **Resource Profiling**: Includes a pre-flight estimator to evaluate computational memory (RAM) and time requirements based on the chosen $k$-mesh before executing heavy matrix diagonalizations.

## Prerequisites

- **Python 3.x**
- `numpy`
- `scipy`
- `matplotlib`

Before running this solver, you need to perform ab-initio DFT calculations (e.g., via VASP or Quantum ESPRESSO) followed by Wannierization using **Wannier90**.

**Required Wannier90 Flags (`wannier90.win`):**
```ini
write_hr = true
write_wsvec = true
position_matrices = true
```

## Setup & Configuration

1. Place the generated Wannier90 files (`_hr.dat`, `_wsvec.dat`, `.wout`) in your working directory.
2. Edit the `tbbse.inp` configuration file to match your system. Example:

```ini
[System]
wannier_prefix = wannier90
a_x = 3.18
a_y = 3.18
c_z = 20.0

[BSE_Parameters]
nk_x = 15
nk_y = 15
kappa = 6.95
r0_ang = 45.0

[Bands]
valence_bands = 0
conduction_bands = 1

[Optics]
polarization = in-plane
broadening = 0.02
energy_range = -0.5, 0.2, 500
```

## Usage

Once configured, simply execute the main solver script:

```bash
python tb_bse_solver.py
```

The script will solve the two-particle eigenvalue problem, print the exciton binding energies, and output the resulting optical absorption spectrum to `tbbse_absorption.png`.

## Documentation

For a comprehensive derivation of the tight-binding BSE theory, non-local screening models, and detailed workflow instructions, please refer to the compiled documentation: **`TBBSE_Documentation.pdf`**.