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
def params_to_T_chol(params, N):
    """
    params: length d^2 real numbers.
      - first d entries -> diagonal (real, >= 0 after abs)
      - remaining -> strictly lower-tri complex entries (row-major)
        order: (1,0), (2,0), (2,1), (3,0), (3,1), (3,2), ...
        each lower entry uses 2 params: real, imag    """


    T = np.zeros((N, N), dtype=complex)

    # diagonal (real, non-negative)
    diag = np.abs(params[:N])
    np.fill_diagonal(T, diag)

    # fill strictly lower triangular entries
    off = params[N:]
    idx = 0
    for r in range(1, N):
        for c in range(r):
            re = off[idx]
            im = off[idx + 1]
            idx += 2
            T[r, c] = re + 1j * im

    return T


def get_rho(nrho=10, N=10, method='HS'):
    
    rho_list = []

    match method:
        case 'HS':
            for _ in range(nrho):
                G = np.random.normal(size=(N,N)) + 1.j * np.random.normal(size=(N,N)) / np.sqrt(2)
                ggt = G @ G.conjugate().T
                rho = ggt / np.trace(ggt)
                rho_list.append(rho)

        case 'MH': # Metropolis-Hastings
            # 1. Cholesky / T-Parametrization
            params = np.random.normal(size=2*N*N)
            T = params_to_T_chol(params, N)
            X = T.conj().T @ T
            rho = X / np.trace(X)
            rho_list.append(rho)


    return rho_list

def likelihood(rho,
               psi_all,
               counts,
               M,
               dx=1
               ):

    lh = 1
    for i in range(M):
        # Extract quadrature eigenstate wavefunction
        psi = psi_all[i]
        # Iterate over histogram bins
        for j, c in enumerate(counts[i]):
            if c == 0:
                continue
            psi_j = psi[:, j][:, np.newaxis]
            p = dx * (psi_j.conj().T @ rho @ psi_j).real.item() 
            if p <= 0:
                p = 1e-15
            lh *= p**c

    return lh

def log_likelihood(rho, psi_all, counts, M, dx=1):
    logL = 0.0
    for i in range(M):
        psi = psi_all[i]
        for j, c in enumerate(counts[i]):
            if c == 0:
                continue
            psi_j = psi[:, j][:, np.newaxis]
            p = dx * (psi_j.conj().T @ rho @ psi_j).real.item()
            if p <= 0:
                p = 1e-15  # avoid log(0)
            logL += c * np.log(p)
    return logL


def accept_rho(rho, rho_new, psi_all, counts, M):
    log_num = log_likelihood(rho=rho_new, psi_all=psi_all, counts=counts, M=M)
    log_den = log_likelihood(rho=rho, psi_all=psi_all, counts=counts, M=M)
    log_A = log_num - log_den
    A = np.min([1, np.exp(log_A)])
    r = np.random.uniform(0,1)
    return r < A
    

def metropolis_hastings(N, psi_all, counts, M, epsilon, max_iter):
    
    # Start with the maximally mixed state
    T = 1/np.sqrt(N) * np.eye(N)
    X = T.conj().T @ T
    rho = X / np.trace(X)

    rho_chain = [rho]

    # Iterate
    print(f"Creating Markov chain...\n")
    for i in tqdm(range(max_iter)):
    #for i in range(max_iter):

        # Create small step (epsilon scale)
        delta_T = np.tril(epsilon * (np.random.normal(size=(N, N)) + 1j * np.random.normal(size=(N, N))))
        T_new = T + delta_T
        X_new = T_new.conj().T @ T_new
        rho_new = X_new / np.trace(X_new)

        # Accept or reject new rho
        if accept_rho(rho, rho_new, psi_all, counts, M):
            rho = rho_new
            T = T_new
        
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

    psi_all = {i: get_overlaps(thetas[i], bin_centers, N) for i in range(M)}

    rho_chain = metropolis_hastings(N,
                                    psi_all,
                                    counts=counts,
                                    M=M,
                                    epsilon=eps,
                                    max_iter=nrho
                                    )
    
    # Estimated rho = avg of Markov chain
    rho_est = 1/len(rho_chain) * np.sum(rho_chain, axis=0)

    return rho_est
    


