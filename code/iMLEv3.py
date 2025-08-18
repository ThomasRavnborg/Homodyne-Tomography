# -*- coding: utf-8 -*-
"""
Created on Fri Aug 15 13:50:36 2025

@author: Thomas Borup Ravnborg
"""
import numpy as np
from utils import get_overlaps, bin_X

def iMLE(thetas, x_vals, N=10, num_bins=150, max_iters=200, tol=1e-3):
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

        # Append log-likelihood value for convergence
        likelihoods.append(logL)
        # Log-likelihood convergence check
        if it > 0 and abs(likelihoods[-1] - likelihoods[-2]) < tol:
            print(f"Converged at iteration {it}, log-likelihood={logL:.6f}")
            break

    print("Warning: Maximum iterations reached without convergence.")

    return rho, likelihoods