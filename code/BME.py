# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 2025

@author: Jan Scarabelli
"""

#%% import necessary libraries

import numpy as np
from pathlib import Path
from tqdm import tqdm
from utils import get_overlaps, bin_X

#%% Define auxiliary functions

def log_likelihood(rho, psi_all, counts, M, dx=1):
    # logL = 0.0
    # for i in range(M):
    #     psi = psi_all[i]
    #     for j, c in enumerate(counts[i]):
    #         if c == 0:
    #             continue
    #         psi_j = psi[:, j][:, np.newaxis]
    #         p = dx * (psi_j.conj().T @ rho @ psi_j).real.item()
    #         if p <= 0:
    #             p = 1e-15  # avoid log(0)
    #         logL += c * np.log(p)
    # return logL

    # Optimised version
    logL = 0.0
    for i in range(M):
        psi = psi_all[i]   # shape (N, num_bins)
        c = counts[i]      # shape (num_bins,)
        rho_psi = rho @ psi             # (N, num_bins)
        p = dx * np.real(np.sum(psi.conj() * rho_psi, axis=0))  # (num_bins,)
        p = np.maximum(p, 1e-15)
        logL += np.dot(c, np.log(p))
    return logL


def accept_rho(rho, rho_new, psi_all, counts, M, logL_old):
    logL_new = log_likelihood(rho_new, psi_all, counts, M)
    logA = logL_new - logL_old
    if np.log(np.random.rand()) < logA:  # log-compare avoids an exp call
        return True, logL_new
    else:
        return False, logL_old

def metropolis_hastings(N, psi_all, counts, M, epsilon, max_iter):
    
    # Start with maximally mixed state
    T = np.eye(N) / np.sqrt(N)
    X = T.conj().T @ T
    rho = X / np.trace(X)

    logL = log_likelihood(rho, psi_all, counts, M)
    rho_chain = [rho]

    print("Computing Markov chain...\n")
    for _ in tqdm(range(max_iter)):
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

            

def bme(thetas,
        x_vals,
        num_bins,
        nrho=10, 
        N=2,
        eps=0.01,
        #max_iter=10000,
        method='HS'):
    
    bin_centers, counts = bin_X(x_vals, num_bins=num_bins)
    M = len(thetas)

    psi_all = [get_overlaps(theta, bin_centers, N) for theta in thetas]

    rho_chain = metropolis_hastings(N, psi_all, counts, M, epsilon=eps, max_iter=nrho)

    burn_in = int(0.2 * len(rho_chain))  # discard first 20%
    samples = rho_chain[burn_in:]

    rho_est = np.mean(samples, axis=0)
    rho_est = (rho_est + rho_est.conj().T) / 2  # enforce Hermitian
    rho_est /= np.trace(rho_est)                # enforce trace 1

    return rho_est
    


