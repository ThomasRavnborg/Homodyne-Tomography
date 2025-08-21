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
    logL_chain = [logL]

    #print("Computing Markov chain...\n")
    for _ in range(nrhos):
        delta_T = np.tril(epsilon * (np.random.normal(size=(N, N)) 
                                     + 1j * np.random.normal(size=(N, N))))
        T_new = T + delta_T
        X_new = T_new.conj().T @ T_new
        rho_new = X_new / np.trace(X_new)

        accepted, logL = accept_rho(rho, rho_new, psi_all, counts, M, logL)
        if accepted:
            rho, T = rho_new, T_new
        
        rho_chain.append(rho)
        logL_chain.append(logL)

    return rho_chain, logL_chain
            

def run_BME(
        thetas: list,
        x_vals: np.ndarray,
        num_bins: int = 200,
        nrho: int = 10000, 
        N: int = 10,
        epsilon: float = 0.01,
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

    rho_chain, logL_chain = metropolis_hastings(psi_all, counts, M=M, N=N, nrhos=nrho, epsilon=epsilon)

    burn_in = int(0.2 * len(rho_chain))  # discard first 20%
    samples = rho_chain[burn_in:]
    logL_subchain = logL_chain[burn_in:]

    rho_est = np.mean(samples, axis=0)
    rho_est = (rho_est + rho_est.conj().T) / 2 
    rho_est /= np.trace(rho_est)  
    
    #logL_avg = np.mean(logL_subchain)             

    return rho_est, logL_subchain
    

#%% 
def run_BME_benchmark(thetas, x_values, N_values, nbin_values, nrhos_values):
    """
    Runs iMLE benchmark for a grid of N and n_bins, 
    returns log-likelihoods and runtimes.
    """
    n_samples = x_values.size  # total number of quadrature measurements

    if len(nrhos_values) == 1:
        likelihood_grid = np.zeros((len(N_values), len(nbin_values)))
        runtime_grid = np.zeros((len(N_values), len(nbin_values)))

        print("Running BME benchmark over N and num. bins...\n")
        n_iter = len(N_values) * len(nbin_values)
        with tqdm(total=n_iter) as pbar:
            for i, N in enumerate(N_values):
                for j, nbins in enumerate(nbin_values):
                    start = time.time()
                    rho_est, logL_subchain = run_BME(thetas, x_values, N=N, num_bins=nbins,
                                    nrho=nrhos_values[0])
                    runtime = time.time() - start

                    likelihood_grid[i, j] = np.mean(logL_subchain)  # take avg log-likelihood
                    runtime_grid[i, j] = runtime

                    pbar.update(1)  # update progress bar
        
        # Normalize likelihood per sample & relative to max
        per_sample = likelihood_grid / n_samples
        delta_ll = per_sample - np.max(per_sample)

    elif len(nrhos_values) > 1:
        likelihood_grid = np.zeros((1, len(nrhos_values)))
        runtime_grid = np.zeros((1, len(nrhos_values)))
        print("Running BME benchmark over num. rhos...\n")
        for k, nrho in tqdm(enumerate(nrhos_values), total=len(nrhos_values)):
            start = time.time()
            #print(f"Running BME for nrho={nrho}")
            rho_est, logL_subchain = run_BME(thetas, x_values, N=N_values[0], num_bins=nbin_values[0],
                                nrho=nrhos_values[k])
            runtime = time.time() - start

            likelihood_grid[0, k] = np.mean(logL_subchain)  # take final log-likelihood
            runtime_grid[0, k] = runtime

        # Normalize likelihood per sample & relative to max
        per_sample = likelihood_grid / n_samples
        delta_ll = per_sample - np.max(per_sample)

    else:
        raise ValueError("nrhos_values must be a list with at least one element.")

    print("BME benchmark completed!\n")

    return delta_ll, runtime_grid
