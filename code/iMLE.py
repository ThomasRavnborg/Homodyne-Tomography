# -*- coding: utf-8 -*-
"""
Created on Fri Aug 15 13:50:36 2025

@author: Thomas Borup Ravnborg
"""
import numpy as np
import time
from tqdm import tqdm
from utils import get_overlaps, bin_X

def run_iMLE(thetas, x_vals, N=10, num_bins=150, max_iters=200, tol=1e-1, run_benchmark=False):
    """
    Iterative Maximum Likelihood Estimation (iMLE).
    
    Inputs:
        thetas : Quadrature phases in radians (ndarray)
        x_vals : Quadrature values (ndarray, shape (M, K))
        N : Fock space cutoff dimension (int)
        num_bins : Number of histogram bins (int)
        max_iters : Maximum number of iterations (int)
        tol : Convergence tolerance for log-likelihood increments (float)
    
    Returns:
        rho : Estimated density matrix (ndarray, shape (N, N))
        likelihoods : list of floats
            Log-likelihood value per iteration
    """
    # Bin quadrature data
    bin_centers, counts = bin_X(x_vals, num_bins=num_bins)
    M = len(thetas)

    # Precompute overlaps psi = <n|x,θ> for all angles and bins
    psi_all = {i: get_overlaps(thetas[i], bin_centers, N) for i in range(M)}

    # Initialize density matrix and likelihood
    rho = np.eye(N, dtype=np.complex128) / N
    likelihoods = []
    rhos = [rho]

    # Iterative algorithm running to maximize log-likelihood
    for it in range(max_iters):
        
        # Initial R and log-likelihood
        R = np.zeros((N, N), dtype=np.complex128)
        logL = 0.0

        # Construct R and calculate log-likelihood over all angles
        for i in range(M):
            # Extract quadrature eigenstate wavefunction
            psi = psi_all[i]
            # Iterate over histogram bins
            for j, c in enumerate(counts[i]):
                if c == 0:
                    continue
                psi_j = psi[:, j][:, np.newaxis]
                p = (psi_j.conj().T @ rho @ psi_j).real.item()
                if p <= 0:
                    p = 1e-15
                R += (c / p) * (psi_j @ psi_j.conj().T)
                logL += c * np.log(p)

        # Update density matrix
        rho = R @ rho @ R
        rho /= np.trace(rho)
        rhos.append(rho)

        # Append log-likelihood value for convergence
        likelihoods.append(logL)
        # Log-likelihood convergence check
        if it > 0 and abs(likelihoods[-1] - likelihoods[-2]) < tol:
            
            if not run_benchmark:
                print(f"Converged at iteration {it}, log-likelihood={logL:.6f}")
                
            break

    #print("Warning: Maximum iterations reached without convergence.")

    return rhos, likelihoods



def run_iMLE_benchmark(thetas, x_values, N_values, nbin_values, max_iters=200, tol=1e-1):
    """
    Runs iMLE benchmark for a grid of N and n_bins, 
    returns log-likelihoods and runtimes.
    """
    n_samples = x_values.size  # total number of quadrature measurements
    likelihood_grid = np.zeros((len(N_values), len(nbin_values)))
    runtime_grid = np.zeros((len(N_values), len(nbin_values)))

    print("Running iMLE benchmark...\n")
    n_iter = len(N_values) * len(nbin_values)
    with tqdm(total=n_iter) as pbar:    
        for i, N in enumerate(N_values):
            for j, nbins in enumerate(nbin_values):
                start = time.time()
                #print(f"Running iMLE for N={N}, bins={nbins}")
                rhos, lls = run_iMLE(thetas, x_values, N=N, num_bins=nbins,
                                max_iters=max_iters, tol=tol, run_benchmark=True)
                runtime = time.time() - start

                likelihood_grid[i, j] = lls[-1]  # take final log-likelihood
                runtime_grid[i, j] = runtime

                pbar.update(1)

    # Normalize likelihood per sample & relative to max
    per_sample = likelihood_grid / n_samples
    delta_ll = per_sample - np.max(per_sample)
    print("MLE benchmark completed!\n")
    
    return delta_ll, runtime_grid

