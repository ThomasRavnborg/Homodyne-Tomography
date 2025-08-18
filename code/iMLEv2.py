# -*- coding: utf-8 -*-
"""
Created on Fri Aug 15 13:50:36 2025

@author: Thomas Borup Ravnborg
"""

import numpy as np
import pandas as pd
from scipy.special import hermite, factorial
from tqdm import tqdm

def get_Pi(theta, x, N):
    """
    Calculates projection operator
    ---------------------------------------------------------
    Inputs:
        theta: Quadrature phase in radians (float)
        x: Quadrature value (float)
        N: Fock space dimension cutoff (int)
    ---------------------------------------------------------
    Returns:
        Pi: ndarray of shape (N, N)
    ---------------------------------------------------------
    """

    # Normalization factor
    norm = (2/np.pi)**(1/4) * np.exp(-x**2)

    # Calculate overlap between fock and quadrature eigenstates
    # for each quadrature point, shape (M, N)
    overlap = np.zeros((N, 1), dtype=np.complex128)
    for n in range(N):
        Hn = hermite(n)(x)
        overlap[n] = norm * Hn / np.sqrt(2**n * factorial(n)) * np.exp(1j * n * theta)

    # Build projection operators, shape (N, N)
    Pi = overlap @ np.conjugate(overlap.T)
    return Pi

def get_Prob(theta, x, N, rho):
    """
    Probability
    ---------------------------------------------------------
    Inputs:
        theta: Quadrature phase in radians (float)
        x: Quadrature value (float)
        N: Fock space dimension cutoff (int)
    ---------------------------------------------------------
    Returns:
        prob: float
    ---------------------------------------------------------
    """

    Pi = get_Pi(theta, x, N)

    return np.trace(Pi @ rho).real

def bin_X(quadratures, num_bins=200, range_x=None):
    """
    Function for binning quadrature values
    ---------------------------------------------------------
    Inputs:
        quadratures: np.array of shape (M, K)
        num_bins: number of histogram bins
        range_x: (min, max) — if None, taken from data
    ---------------------------------------------------------
    Returns:
        bin_centers: shape (num_bins,)
        counts: shape (M, num_bins)
    ---------------------------------------------------------
    """
    M, K = quadratures.shape
    if range_x is None:
        min_x, max_x = np.min(quadratures), np.max(quadratures)
    else:
        min_x, max_x = range_x

    counts = np.zeros((M, num_bins), dtype=int)
    bin_edges = np.linspace(min_x, max_x, num_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    for i in range(M):
        counts[i], _ = np.histogram(quadratures[i], bins=bin_edges)

    return bin_centers, counts


def MaximumLikelihood(thetas, x_values, N=10, num_bins=150, max_iters=200):

    bin_centers, counts = bin_X(x_values, num_bins=num_bins)

    rho = np.eye(N) / N  # Initial guess for the density matrix

    for it in tqdm(range(max_iters)):

        R = np.zeros((N, N), dtype=complex)

        for th in thetas:
            for bin in bin_centers:
                Pi = get_Pi(th, bin, N)
                Pr = get_Prob(th, bin, N, rho)
                if Pr <= 0:
                    Pr = 1e-15
                R += counts[th, bin]/Pr * Pi

        rho = R @ rho @ R
        rho /= np.trace(rho)


def iMLE(thetas, x_values, N=10, num_bins=150, max_iters=50, tol=1e-6):
    """
    Iterative Maximum Likelihood Estimation (iMLE) for
    quantum state tomography.
    ---------------------------------------------------------
    Inputs:
        thetas: Array of quadrature phases (1D array)
        x_values: Array of quadrature values (2D array)
        N: Fock space dimension cutoff (int)
        num_bins: Number of histogram bins (int)
        max_iters: Maximum number of iterations (int)
        tol: Convergence tolerance (float)
    ---------------------------------------------------------
    Returns:
        rho: Estimated density matrix (ndarray)
    ---------------------------------------------------------
    """
    bin_centers, counts = bin_X(x_values, num_bins=num_bins)
    #num_bins = len(bin_centers)
    M = len(thetas)

    psi_all = {}
    for i in range(M):
        psi_all[i] = Pi_mn(thetas[i], bin_centers, N)  # (N, num_bins)

    rho = np.eye(N) / N

    for it in tqdm(range(max_iters)):
        R = np.zeros((N, N), dtype=complex)

        for i in range(M):
            psi = psi_all[i]
            for j in range(num_bins):
                c = counts[i, j]
                if c == 0:
                    continue
                psi_j = psi[:, j][:, np.newaxis]
                p = (psi_j.conj().T @ rho @ psi_j).real.item()
                if p <= 0:
                    p = 1e-15
                R += c * (psi_j @ psi_j.conj().T) / p

        rho_new = R @ rho @ R
        rho_new /= np.trace(rho_new)

        if np.linalg.norm(rho_new - rho) < tol:
            print(f"Converged in {it} iterations.")
            break
        rho = rho_new
    return rho


import os
from pathlib import Path
# Loading data
parent = Path(os.path.dirname(os.getcwd()))
date = "091027"
state = "tora12"

data_path = parent / "Homodyne-Tomograpy" / "data" / "dataframes" / date
print(data_path)
data = pd.read_csv(data_path / (state + '.csv'))
# Remove first column
data = data.iloc[:, 1:]
data
# Convert to numpy array
x_values = np.array(data)
x_values = np.swapaxes(x_values, 0, 1)

bin_X(x_values, num_bins=200, range_x=None)


# Make array from 0 to 165 in steps of 15
thetas = np.arange(0, 166, 15)
# Convert to radians
thetas = np.radians(thetas)
theta0 = np.deg2rad(55)
x0 = -0.1

theta0 = 0
x0 = 0

rho_est = MaximumLikelihood(thetas-theta0, x_values-x0, N=20, num_bins=200, max_iters=100)

print(rho_est)