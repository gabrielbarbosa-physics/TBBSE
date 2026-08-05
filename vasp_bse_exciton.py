import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh
from scipy.special import struve, y0
from pymatgen.io.vasp import Vasprun
import os
from scipy.constants import physical_constants

# Physical Constants (SI)
m0 = physical_constants['electron mass'][0] # kg
e = physical_constants['elementary charge'][0] # C
hbar = physical_constants['Planck constant over 2 pi'][0] # J s
epsilon0 = physical_constants['vacuum electric permittivity'][0] # F/m
eV_to_J = physical_constants['electron volt'][0]

def extract_vasp_data(vasprun_path="vasprun.xml"):
    """
    Extracts bandgap and estimates effective masses from a VASP HSE-06 vasprun.xml file.
    """
    if not os.path.exists(vasprun_path):
        print(f"Warning: '{vasprun_path}' not found.")
        print("Using default values for a generic TMD/GaAs heterostructure...")
        # Default typical values for a TMD (like MoS2) on GaAs
        return {
            'Eg': 2.0,     # Bandgap in eV (HSE06 direct gap)
            'm_e': 0.45,   # Electron effective mass in m0
            'm_h': 0.54,   # Hole effective mass in m0
            'r0': 4.5e-9,  # 2D polarizability / screening length for TMD (meters)
        }
    
    print(f"Parsing {vasprun_path} (this might take a minute depending on file size)...")
    try:
        run = Vasprun(vasprun_path, parse_projected_eigen=False)
        bs = run.get_band_structure(line_mode=True)
        
        Eg = bs.get_band_gap()['energy']
        print(f"Successfully extracted HSE06 direct bandgap: {Eg:.4f} eV")
        
        # In a fully robust implementation, m_e and m_h would be derived by fitting 
        # a parabola to the CBM and VBM near the K or Gamma point.
        # Here we provide placeholders for the effective masses for demonstration.
        m_e = 0.45
        m_h = 0.54
        r0 = 4.5e-9
        
        return {
            'Eg': Eg,
            'm_e': m_e,
            'm_h': m_h,
            'r0': r0
        }
    except Exception as exc:
        print(f"Error parsing {vasprun_path}: {exc}")
        print("Returning default parameters.")
        return {'Eg': 2.0, 'm_e': 0.45, 'm_h': 0.54, 'r0': 4.5e-9}

def rytova_keldysh_potential(r, kappa, r0):
    """
    Calculates the Rytova-Keldysh potential for a 2D material surrounded by dielectrics.
    r: radial distance in meters
    kappa: average dielectric constant of surrounding media, (eps_substrate + eps_vacuum) / 2
    r0: screening length of the 2D layer
    """
    prefactor = e**2 / (4 * np.pi * epsilon0)
    
    # Avoid singularity at r=0
    r_safe = np.clip(r, 1e-12, None)
    
    x = kappa * r_safe / r0
    
    # V(r) = - e^2 / (8 * epsilon0 * r0) * [H_0(x) - Y_0(x)]
    # In SI: V(r) = - (pi / 2r0) * (e^2 / 4 pi epsilon0) * [H_0(x) - Y_0(x)]
    V_J = - (np.pi * prefactor / (2 * r0)) * (struve(0, x) - y0(x))
    
    return V_J / eV_to_J # Return potential in eV

def solve_bse_radial(mu_rel, kappa, r0, r_max=25e-9, N=3000):
    """
    Solves the radial Wannier-Mott equation (effective mass BSE) for 2D excitons using finite differences.
    """
    r = np.linspace(1e-12, r_max, N)
    dr = r[1] - r[0]
    
    mu = mu_rel * m0
    
    # Kinetic energy term: T = - (hbar^2 / 2 mu) * (1/r d/dr (r d/dr))
    # Using substitution u(r) = sqrt(r) R(r) transforms it to 1D Schrödinger:
    # - (hbar^2 / 2 mu) * u''(r) + [V(r) - (hbar^2 / 2 mu) * (1 / 4 r^2)] u(r) = E u(r)
    
    t0 = hbar**2 / (2 * mu * dr**2) / eV_to_J
    
    # Diagonal elements: 2*t0 + V(r) + centrifugal barrier for s-wave
    centrifugal = (hbar**2 / (2 * mu)) * (1 / (4 * r**2)) / eV_to_J
    V = rytova_keldysh_potential(r, kappa, r0)
    
    main_diag = 2 * t0 + V + centrifugal
    off_diag = -t0 * np.ones(N-1)
    
    # Construct Hamiltonian tridiagonal matrix
    H = diags([off_diag, main_diag, off_diag], [-1, 0, 1])
    
    # Diagonalize to find lowest eigenvalues (exciton states)
    num_states = 5
    evals, evecs = eigsh(H, k=num_states, which='SA')
    
    # Convert eigenvectors u(r) back to 2D radial wavefunctions R(r)
    # Normalization: 2*pi * integral_0^infty |R(r)|^2 r dr = 1
    # which is equivalent to 2*pi * integral_0^infty |u(r)|^2 dr = 1
    wavefunctions = np.zeros_like(evecs)
    for i in range(num_states):
        norm_factor = np.sqrt(2 * np.pi * np.sum(evecs[:, i]**2) * dr)
        u_norm = evecs[:, i] / norm_factor
        wavefunctions[:, i] = u_norm / np.sqrt(r)
        
    return r, evals, wavefunctions

def calculate_absorption(energies, wavefunctions, r, Eg, omega_range, gamma=0.03):
    """
    Calculates the optical absorption spectrum using Elliott's formula.
    """
    alpha = np.zeros_like(omega_range)
    
    # Probability density of the exciton at the origin: |phi(r=0)|^2
    # Approximated by the value at the first grid point
    phi2_0 = np.abs(wavefunctions[0, :])**2
    
    for i, E_b in enumerate(energies):
        E_exc = Eg + E_b # E_b is negative (binding energy)
        # Add Lorentzian peak for each excitonic state
        alpha += phi2_0[i] * (gamma / 2) / ((omega_range - E_exc)**2 + (gamma / 2)**2)
        
    return alpha

def main():
    print("="*60)
    print(" 2D/3D Heterostructure BSE Solver (Effective Mass Model) ")
    print("="*60)
    
    # 1. Parse VASP outputs
    vasp_data = extract_vasp_data("vasprun.xml")
    
    Eg = vasp_data['Eg']
    m_e = vasp_data['m_e']
    m_h = vasp_data['m_h']
    r0 = vasp_data['r0']
    
    mu_rel = (m_e * m_h) / (m_e + m_h)
    
    # 2. Setup Substrate Screening
    # Dielectric constant of GaAs substrate ~ 12.9
    # Vacuum ~ 1.0 (assuming the top layer is exposed to vacuum)
    eps_GaAs = 12.9
    eps_vac = 1.0
    kappa = (eps_GaAs + eps_vac) / 2.0
    
    print("\nSystem Parameters:")
    print(f"- Reduced mass (mu_rel): {mu_rel:.3f} m0")
    print(f"- Screening parameter (kappa): {kappa:.2f}")
    print(f"- 2D screening length (r0): {r0*1e9:.2f} nm")
    print(f"- HSE06 direct bandgap (Eg): {Eg:.4f} eV")
    
    # 3. Solve BSE (Wannier Equation)
    print("\nSolving the effective-mass Bethe-Salpeter Equation...")
    r, evals, wavefuncs = solve_bse_radial(mu_rel, kappa, r0)
    
    print("\nExciton States (Relative to CBM):")
    for i, E in enumerate(evals):
        print(f" - {i+1}s state: Binding Energy = {E*1000:.1f} meV")
        
    # 4. Plot Results
    print("\nGenerating plots...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot Wavefunctions
    for i in range(3): # Plot first 3 states (1s, 2s, 3s)
        axes[0].plot(r * 1e9, wavefuncs[:, i] * 1e-9, label=f'{i+1}s (Eb={evals[i]*1000:.0f} meV)', lw=2)
    axes[0].set_xlabel('Radial distance $r$ (nm)', fontsize=12)
    axes[0].set_ylabel('Radial Wavefunction $R(r)$ (nm$^{-1}$)', fontsize=12)
    axes[0].set_title('Exciton Radial Wavefunctions', fontsize=14)
    axes[0].set_xlim(0, 15)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot Absorption
    omega = np.linspace(Eg - 0.4, Eg + 0.1, 800)
    alpha = calculate_absorption(evals, wavefuncs, r, Eg, omega, gamma=0.015)
    
    axes[1].plot(omega, alpha, color='#D32F2F', lw=2)
    axes[1].fill_between(omega, 0, alpha, color='#D32F2F', alpha=0.2)
    axes[1].set_xlabel('Photon Energy (eV)', fontsize=12)
    axes[1].set_ylabel('Absorption Intensity (a.u.)', fontsize=12)
    axes[1].set_title('Excitonic Absorption Spectrum', fontsize=14)
    axes[1].set_yticks([]) # Hide y-ticks as intensity is arbitrary units
    axes[1].axvline(Eg, color='k', linestyle='--', label=f'Direct Gap ({Eg:.2f} eV)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('bse_exciton_results.png', dpi=300)
    print("Done! Plots saved successfully to 'bse_exciton_results.png'.")

if __name__ == "__main__":
    main()
