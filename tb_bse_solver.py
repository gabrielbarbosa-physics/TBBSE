import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import configparser
import time
from scipy.constants import physical_constants

# ==============================================================================
# Physical Constants (SI Units)
# ==============================================================================
eV_to_J = physical_constants['electron volt'][0]
e = physical_constants['elementary charge'][0]
epsilon0 = physical_constants['vacuum electric permittivity'][0]
hbar = physical_constants['Planck constant over 2 pi'][0]
m0 = physical_constants['electron mass'][0]

def parse_input_script(filename="tbbse.inp"):
    """
    Parses the input configuration file (e.g., tbbse.inp).
    This allows the user to configure the run without modifying the Python script.
    Provides robust default values if the file or specific options are missing.
    """
    config = configparser.ConfigParser()
    if not os.path.exists(filename):
        print(f"Warning: Input script '{filename}' not found. Using default parameters.")
        config.read_dict({
            'System': {'wannier_prefix': 'wannier90', 'a_x': '3.18', 'a_y': '3.18', 'c_z': '20.0'},
            'BSE_Parameters': {'nk_x': '15', 'nk_y': '15', 'kappa': '6.95', 'r0_ang': '45.0'},
            'Bands': {'valence_bands': '0', 'conduction_bands': '1'},
            'Optics': {'polarization': 'in-plane', 'broadening': '0.02', 'energy_range': '-0.5, 0.2, 500'}
        })
    else:
        config.read(filename)
        
    # Extract values into a parameter dictionary
    params = {}
    params['prefix'] = config.get('System', 'wannier_prefix', fallback='wannier90')
    params['a_x'] = config.getfloat('System', 'a_x', fallback=3.18) * 1e-10
    params['a_y'] = config.getfloat('System', 'a_y', fallback=3.18) * 1e-10
    params['c_z'] = config.getfloat('System', 'c_z', fallback=20.0) * 1e-10
    
    params['nk_x'] = config.getint('BSE_Parameters', 'nk_x', fallback=15)
    params['nk_y'] = config.getint('BSE_Parameters', 'nk_y', fallback=15)
    params['kappa'] = config.getfloat('BSE_Parameters', 'kappa', fallback=6.95)
    params['r0'] = config.getfloat('BSE_Parameters', 'r0_ang', fallback=45.0) * 1e-10
    
    params['v_idx'] = config.getint('Bands', 'valence_bands', fallback=0)
    params['c_idx'] = config.getint('Bands', 'conduction_bands', fallback=1)
    
    params['polarization'] = config.get('Optics', 'polarization', fallback='in-plane')
    params['gamma'] = config.getfloat('Optics', 'broadening', fallback=0.02)
    
    er_str = config.get('Optics', 'energy_range', fallback='-0.5, 0.2, 500')
    er_parts = [float(x) for x in er_str.split(',')]
    params['E_min'] = er_parts[0]
    params['E_max'] = er_parts[1]
    params['nE'] = int(er_parts[2])
    
    return params

def estimate_computational_cost(nk_x, nk_y, num_v, num_c):
    """
    Estimates the memory and computational time required for the BSE diagonalization.
    The BSE Hamiltonian is a dense complex matrix of rank N = (nk_x * nk_y) * num_v * num_c.
    Diagonalization via LAPACK (numpy.linalg.eigh) scales as O(N^3).
    """
    N_k = nk_x * nk_y
    N_bse = N_k * num_v * num_c
    
    # A complex128 number uses 16 bytes. A dense NxN matrix uses N^2 * 16 bytes.
    mem_bytes = N_bse**2 * 16
    mem_gb = mem_bytes / (1024**3)
    
    # Rough time estimate: on a standard modern CPU, solving a 2000x2000 complex eigh takes ~2-3 seconds.
    # We scale this using the O(N^3) law.
    time_sec = (N_bse / 2000.0)**3 * 2.5
    
    print("\n" + "="*50)
    print(" HARDWARE PRE-REQUISITES & COST ESTIMATION")
    print("="*50)
    print(f"BSE Matrix Rank: {N_bse} x {N_bse}")
    print(f"Estimated Peak Memory for Matrix: {mem_gb:.2f} GB")
    
    if time_sec < 60:
        print(f"Estimated Diagonalization Time: {time_sec:.1f} seconds")
    elif time_sec < 3600:
        print(f"Estimated Diagonalization Time: {time_sec/60:.1f} minutes")
    else:
        print(f"Estimated Diagonalization Time: {time_sec/3600:.1f} hours")
        
    print("="*50)
    
    if mem_gb > 16.0:
        print("WARNING: Matrix requires > 16 GB RAM. Consider reducing nk_x/nk_y or using an HPC node.")
    if time_sec > 3600:
        print("WARNING: Diagonalization will take > 1 hour. This script runs sequentially.")
    print("")

def parse_wannier90_hr(filename):
    """
    Parses the standard wannier90_hr.dat file to extract tight-binding hopping parameters H(R).
    This matrix describes the electronic band structure in the real-space Wannier basis.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Missing {filename}. Please run Wannier90 first.")
        
    with open(filename, 'r') as f:
        lines = f.readlines()
        
    num_wann = int(lines[1].strip())
    nrpts = int(lines[2].strip())
    
    # Extract the degeneracy (weighting) of each Wigner-Seitz grid point
    deg = []
    idx = 3
    while len(deg) < nrpts:
        deg.extend([int(x) for x in lines[idx].split()])
        idx += 1
        
    R_vectors = []
    H_R = np.zeros((nrpts, num_wann, num_wann), dtype=complex)
    
    for i in range(nrpts):
        for m in range(num_wann):
            for n in range(num_wann):
                parts = lines[idx].split()
                if m == 0 and n == 0:
                    R_vectors.append([int(parts[0]), int(parts[1]), int(parts[2])])
                real_part = float(parts[5])
                imag_part = float(parts[6])
                H_R[i, m, n] = real_part + 1j * imag_part
                idx += 1
                
    return num_wann, nrpts, np.array(R_vectors), H_R, np.array(deg)

def parse_wannier90_wsvec(filename):
    """
    Parses wannier90_wsvec.dat.
    This file explicitly lists the degenerate Wigner-Seitz vectors connecting two unit cells, 
    allowing us to strictly preserve the spatial symmetries during the Fourier Transform.
    """
    if not os.path.exists(filename):
        print(f"Notice: {filename} not found. Will use standard R vectors and degeneracies from hr.dat.")
        return None
        
    with open(filename, 'r') as f:
        lines = f.readlines()
        
    wsvecs = []
    idx = 1
    while idx < len(lines):
        line = lines[idx].strip()
        if not line:
            idx += 1
            continue
        parts = line.split()
        if len(parts) == 3:
            # We found a new R-vector block
            idx += 1
            ndeg = int(lines[idx].strip()) # Number of degenerate vectors
            idx += 1
            vectors = []
            for _ in range(ndeg):
                vparts = lines[idx].split()
                vectors.append([float(vparts[0]), float(vparts[1]), float(vparts[2])])
                idx += 1
            wsvecs.append(np.array(vectors))
        else:
            idx += 1
    return wsvecs

def parse_wannier90_centres(prefix):
    """
    Extracts the Wannier centres tau_m from wannier90.wout.
    These centres are used to apply a sub-cell phase factor (gauge correction) 
    to the macroscopic Coulomb interaction W(q).
    """
    wout_file = f"{prefix}.wout"
    if not os.path.exists(wout_file):
        print(f"Notice: {wout_file} not found. Cannot extract Wannier centres. Gauge phases in W(q) will be neglected.")
        return None
        
    with open(wout_file, 'r') as f:
        lines = f.readlines()
        
    centres = []
    in_final_state = False
    for line in lines:
        if "Final State" in line:
            in_final_state = True
            centres = []
        # Look for lines containing "WF centre and spread" in the final state block
        if in_final_state and "WF centre and spread" in line:
            parts = line.split('(')[1].split(')')[0].split(',')
            centres.append([float(parts[0]), float(parts[1]), float(parts[2])])
        if in_final_state and "Sum of centres and spreads" in line:
            break
            
    if not centres:
        return None
        
    # Return centres as a numpy array in Angstroms
    return np.array(centres)

def build_hk_and_vk(k_point, num_wann, nrpts, R_vectors, H_R, deg, wsvecs=None, lattice_vectors=None):
    """
    Constructs the reciprocal space Hamiltonian H(k) and velocity matrix elements v(k).
    Velocity v(k) is calculated analytically via Peierls substitution: v(k) = (1/hbar) * dH(k)/dk.
    """
    H_k = np.zeros((num_wann, num_wann), dtype=complex)
    
    # Velocity operators v_x, v_y (in units of eV * Angstrom, will convert later)
    v_k = np.zeros((2, num_wann, num_wann), dtype=complex) 
    
    if wsvecs is not None:
        # 1. Use exact Wigner-Seitz vectors (from wannier90_wsvec.dat)
        # H(k) = sum_R (1/N_deg) sum_{r_ws} e^{i k . r_ws} H(R)
        for i in range(nrpts):
            R_ws_list = wsvecs[i]
            ndeg = len(R_ws_list)
            
            phase_factor = 0.0 + 0.0j
            dphase_dkx = 0.0 + 0.0j
            dphase_dky = 0.0 + 0.0j
            
            for r_ws in R_ws_list:
                # r_ws is in fractional coordinates. k . r_ws requires 2*pi
                kr = 2.0 * np.pi * np.dot(k_point, r_ws)
                exp_ikr = np.exp(1j * kr)
                phase_factor += exp_ikr
                
                if lattice_vectors is not None:
                    # Convert fractional distance to Cartesian Angstroms
                    r_cart = np.dot(r_ws, lattice_vectors)
                    dphase_dkx += 1j * r_cart[0] * exp_ikr
                    dphase_dky += 1j * r_cart[1] * exp_ikr
                    
            phase_factor /= ndeg
            dphase_dkx /= ndeg
            dphase_dky /= ndeg
            
            H_k += phase_factor * H_R[i]
            v_k[0] += dphase_dkx * H_R[i]
            v_k[1] += dphase_dky * H_R[i]
            
    else:
        # 2. Use simple R vectors and degeneracy (from wannier90_hr.dat)
        for i in range(nrpts):
            R = R_vectors[i]
            kr = 2.0 * np.pi * np.dot(k_point, R)
            exp_ikr = np.exp(1j * kr) / deg[i]
            
            H_k += exp_ikr * H_R[i]
            
            if lattice_vectors is not None:
                R_cart = np.dot(R, lattice_vectors)
                v_k[0] += 1j * R_cart[0] * exp_ikr * H_R[i]
                v_k[1] += 1j * R_cart[1] * exp_ikr * H_R[i]
                
    return H_k, v_k

def rytova_keldysh_q(q_mag, kappa, r0):
    """
    Calculates the macroscopic screened Coulomb interaction W(q) in momentum space
    using the Rytova-Keldysh model.
    This model captures the non-local screening effect inherent to 2D materials (e.g. MX2).
    """
    if q_mag < 1e-6:
        # q=0 singularity handled via analytical cell integration in a full grid approach, 
        # or ignored here as a minimal cutoff.
        return 0.0 
        
    prefactor = e**2 / (2 * epsilon0)
    W_J = prefactor / (q_mag * (kappa + r0 * q_mag))
    
    return W_J / eV_to_J # Returns W in eV * m^2

def solve_tb_bse(params):
    """
    Builds and diagonalizes the two-particle Bethe-Salpeter Equation (BSE) matrix 
    using the extracted tight-binding properties.
    """
    prefix = params['prefix']
    print(f"Parsing Wannier90 Hamiltonian ({prefix}_hr.dat)...")
    num_wann, nrpts, R_vectors, H_R, deg = parse_wannier90_hr(f"{prefix}_hr.dat")
    
    wsvecs = parse_wannier90_wsvec(f"{prefix}_wsvec.dat")
    centres = parse_wannier90_centres(prefix)
    
    a_x, a_y, c_z = params['a_x'], params['a_y'], params['c_z']
    
    # Define Cartesian lattice vectors (in Angstroms) for the derivative operator
    lattice_vectors = np.array([
        [a_x*1e10, 0, 0], 
        [0, a_y*1e10, 0], 
        [0, 0, c_z*1e10]
    ]) 
    
    nk_x, nk_y = params['nk_x'], params['nk_y']
    
    # Uniform mesh generation over the 2D Brillouin Zone
    k_mesh = []
    for i in range(nk_x):
        for j in range(nk_y):
            k_mesh.append([i/nk_x, j/nk_y, 0.0])
    k_mesh = np.array(k_mesh)
    N_k = len(k_mesh)
    
    v_idx = params['v_idx']
    c_idx = params['c_idx']
    
    # Estimate resource requirements before beginning heavy computations
    estimate_computational_cost(nk_x, nk_y, 1, 1) # using 1 V-band and 1 C-band
    
    # -------------------------------------------------------------------------
    # Step 1: Diagonalize the single-particle Hamiltonian H(k)
    # -------------------------------------------------------------------------
    print(f"Solving single-particle H(k) on {nk_x}x{nk_y} grid ({N_k} k-points)...")
    energies = np.zeros((N_k, num_wann))
    eigenvectors = np.zeros((N_k, num_wann, num_wann), dtype=complex)
    
    # Arrays to store optical transition matrix elements for later absorption calc
    optical_mat_x = np.zeros(N_k, dtype=complex)
    optical_mat_y = np.zeros(N_k, dtype=complex)
    
    for i, k in enumerate(k_mesh):
        H_k, v_k = build_hk_and_vk(k, num_wann, nrpts, R_vectors, H_R, deg, wsvecs, lattice_vectors)
        
        # Diagonalize Hermitian H(k)
        evals, evecs = np.linalg.eigh(H_k)
        energies[i] = evals
        eigenvectors[i] = evecs
        
        # Velocity matrix elements <c | v | v> = sum_{n,m} c_c,n^* v_{n,m} c_v,m
        # Peierls derivative dH/dk acts as our unscaled velocity operator.
        optical_mat_x[i] = np.vdot(evecs[:, c_idx], np.dot(v_k[0], evecs[:, v_idx]))
        optical_mat_y[i] = np.vdot(evecs[:, c_idx], np.dot(v_k[1], evecs[:, v_idx]))
        
    # -------------------------------------------------------------------------
    # Step 2: Construct the two-particle BSE matrix
    # -------------------------------------------------------------------------
    print("Constructing BSE two-particle Hamiltonian...")
    H_BSE = np.zeros((N_k, N_k), dtype=complex)
    kappa = params['kappa']
    r0 = params['r0']
    Area = a_x * a_y # Unit cell area in m^2
    
    for i in range(N_k):
        # Diagonal term: The kinetic energy of the non-interacting electron-hole pair
        H_BSE[i, i] = energies[i, c_idx] - energies[i, v_idx]
        
        for j in range(N_k):
            if i == j: continue
            
            # Momentum transfer q = k_i - k_j
            dq = k_mesh[i] - k_mesh[j]
            dq -= np.round(dq) # Fold back into the 1st BZ
            
            # Convert reciprocal fractional coordinates to Cartesian (1/m)
            b1 = 2 * np.pi / a_x * np.array([1, 0, 0])
            b2 = 2 * np.pi / a_y * np.array([0, 1, 0])
            q_cart = dq[0] * b1 + dq[1] * b2
            q_mag = np.linalg.norm(q_cart)
            
            # Evaluate macroscopic interaction W(q)
            W_q = rytova_keldysh_q(q_mag, kappa, r0) / Area
            
            # Sub-cell gauge factor if Wannier centers are provided
            gauge_factor = 1.0 + 0.0j
            if centres is not None:
                # This phase term captures the relative displacement of the V and C 
                # Wannier orbitals within the unit cell, improving sub-cell accuracy.
                tau_c = centres[c_idx] * 1e-10 # Convert to meters
                tau_v = centres[v_idx] * 1e-10
                phase = np.exp(1j * np.dot(q_cart, tau_c - tau_v))
                gauge_factor = phase
            
            # Compute wavefunction overlap <v_i | v_j> and <c_j | c_i>
            overlap_v = np.vdot(eigenvectors[i, :, v_idx], eigenvectors[j, :, v_idx])
            overlap_c = np.vdot(eigenvectors[j, :, c_idx], eigenvectors[i, :, c_idx])
            
            # Assemble Matrix Element (excluding exchange for pure singlet formulation)
            H_BSE[i, j] -= W_q * overlap_v * overlap_c * gauge_factor

    # -------------------------------------------------------------------------
    # Step 3: Diagonalize the BSE matrix to obtain excitonic states
    # -------------------------------------------------------------------------
    print("Diagonalizing BSE matrix...")
    t_start = time.time()
    bse_evals, bse_evecs = np.linalg.eigh(H_BSE)
    print(f"Diagonalization finished in {time.time()-t_start:.1f} seconds.")
    
    min_gap = np.min(energies[:, c_idx] - energies[:, v_idx])
    print(f"\nMinimum Quasiparticle Gap (from TB): {min_gap:.4f} eV")
    print(f"Exciton Binding Energies (top 5 strongly bound states):")
    for i in range(min(5, N_k)):
        Eb = min_gap - bse_evals[i]
        print(f" State {i+1}: Eb = {Eb*1000:.2f} meV (Absolute Energy = {bse_evals[i]:.4f} eV)")
        
    return bse_evals, bse_evecs, min_gap, optical_mat_x, optical_mat_y

def main():
    print("="*60)
    print(" Tight-Binding Bethe-Salpeter Equation (TB-BSE) Solver ")
    print("="*60)
    
    # 1. Parse configuration from the input file
    params = parse_input_script("tbbse.inp")
    
    # 2. Build and solve the BSE
    bse_evals, bse_evecs, min_gap, opt_x, opt_y = solve_tb_bse(params)
    
    # 3. Calculate Optical Absorption Spectrum using Elliott's Formula
    print("\nCalculating optical absorption spectrum...")
    omega = np.linspace(min_gap + params['E_min'], min_gap + params['E_max'], params['nE'])
    alpha = np.zeros_like(omega)
    gamma = params['gamma']
    
    pol = params['polarization'].lower()
    
    for idx, E_exc in enumerate(bse_evals):
        # The exciton wavefunction in k-space A_k
        A_k = bse_evecs[:, idx]
        
        # Coherent sum over k-space incorporating the optical transition matrix element.
        # This gives the correct oscillator strength (optical selection rules).
        Mx = np.abs(np.sum(A_k * opt_x))**2
        My = np.abs(np.sum(A_k * opt_y))**2
        
        if pol == 'x':
            osc_strength = Mx
        elif pol == 'y':
            osc_strength = My
        else: # average for 'in-plane'
            osc_strength = 0.5 * (Mx + My)
            
        # Add Lorentzian broadening for this specific exciton peak
        alpha += osc_strength * (gamma / 2) / ((omega - E_exc)**2 + (gamma / 2)**2)
        
    # 4. Plotting the results
    plt.figure(figsize=(8, 5))
    plt.plot(omega, alpha, color='#1976D2', lw=2)
    plt.fill_between(omega, 0, alpha, color='#1976D2', alpha=0.3)
    plt.axvline(min_gap, color='k', linestyle='--', label=f'TB Quasiparticle Gap ({min_gap:.2f} eV)')
    
    plt.title(f'TB-BSE Excitonic Absorption Spectrum (Pol: {pol})', fontsize=14)
    plt.xlabel('Photon Energy (eV)', fontsize=12)
    plt.ylabel('Absorption Intensity (Arbitrary Units)', fontsize=12)
    plt.yticks([])
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('tbbse_absorption.png', dpi=300)
    print("Absorption spectrum successfully plotted and saved to 'tbbse_absorption.png'.")

if __name__ == "__main__":
    main()
