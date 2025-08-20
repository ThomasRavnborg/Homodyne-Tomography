# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 2025

@author: Jan Scarabelli
"""

#%% import necessary libraries

import numpy as np
import time
from pathlib import Path
from tqdm import tqdm
from utils import get_overlaps, bin_X, log_likelihood, accept_rho

#%% Define function to apply Metropolis-Hastings method
def metropolis_hastings(
        psi_all: list, 
        counts: list,
        M: int,
        N: int = 10,
        nrhos: int = 10000,
        epsilon: float = 0.01,
        ):
    """
    Metropolis-Hastings algorithm to sample density matrices.

    Inputs:
        N : Fock space cutoff dimension (int)
        psi_all : Overlap vectors for all angles (list of ndarrays)
        counts : Histogram counts for each angle (ndarray, shape (M, num_bins))
        M : Number of angles (int)
        epsilon : Step size for perturbation (float)
        nrhos : Maximum number of iterations (int)

    Returns:
        rho_chain : List of sampled density matrices (list of ndarrays)
    """
    # Start with maximally mixed state
    T = np.eye(N) / np.sqrt(N)
    X = T.conj().T @ T
    rho = X / np.trace(X)

    logL = log_likelihood(rho, psi_all, counts, M)
    rho_chain = [rho]

    print("Computing Markov chain...\n")
    for _ in tqdm(range(nrhos)):
        delta_T = np.tril(epsilon * (np.random.normal(size=(N, N)) 
                                     + 1j * np.random.normal(size=(N, N))))
        T_new = T + delta_T
        X_new = T_new.conj().T @ T_new
        rho_new = X_new / np.trace(X_new)

        accepted, logL = accept_rho(rho, rho_new, psi_all, counts, M, logL)
        if accepted:
            rho, T = rho_new, T_new
        rho_chain.append(rho)

    return rho_chain
            

def run_BME(
        thetas: list,
        x_vals: np.ndarray,
        num_bins: int = 200,
        nrho: int = 10000, 
        N: int = 10,
        epsilon: float =0.01,
        ):
    """
    Runs Bayesian Maximum Entropy (BME) estimation using Metropolis-Hastings.

    Inputs:
        thetas : Quadrature phases in radians (list of floats)
        x_vals : Quadrature values (ndarray, shape (M, K))
        num_bins : Number of histogram bins (int)
        nrho : Number of Metropolis-Hastings iterations (int)
        N : Fock space cutoff dimension (int)
        epsilon : Step size for perturbation (float)
        
    Returns:
        rho_est : Estimated density matrix (ndarray, shape (N, N))
    """
    
    bin_centers, counts = bin_X(x_vals, num_bins=num_bins)
    M = len(thetas)

    psi_all = [get_overlaps(theta, bin_centers, N) for theta in thetas]

    rho_chain = metropolis_hastings(N, psi_all, counts, M, epsilon=epsilon, max_iter=nrho)

    burn_in = int(0.2 * len(rho_chain))  # discard first 20%
    samples = rho_chain[burn_in:]

    rho_est = np.mean(samples, axis=0)
    rho_est = (rho_est + rho_est.conj().T) / 2  # enforce Hermitian
    rho_est /= np.trace(rho_est)                # enforce trace 1

    return rho_est
    

#%% 
def run_BME_benchmark(thetas, x_values, N_values, nbin_values, max_iters=200, tol=1e-1):
    """
    Runs iMLE benchmark for a grid of N and n_bins, 
    returns log-likelihoods and runtimes.
    """
    n_samples = x_values.size  # total number of quadrature measurements
    likelihood_grid = np.zeros((len(N_values), len(nbin_values)))
    runtime_grid = np.zeros((len(N_values), len(nbin_values)))

    for i, N in enumerate(N_values):
        for j, nbins in enumerate(nbin_values):
            start = time.time()
            print(f"Running iMLE for N={N}, bins={nbins}")
            rhos, lls = run_BME(thetas, x_values, N=N, num_bins=nbins,
                             max_iters=max_iters, tol=tol)
            runtime = time.time() - start

            likelihood_grid[i, j] = lls[-1]  # take final log-likelihood
            runtime_grid[i, j] = runtime

    # Normalize likelihood per sample & relative to max
    per_sample = likelihood_grid / n_samples
    delta_ll = per_sample - np.max(per_sample)

    return delta_ll, runtime_grid
