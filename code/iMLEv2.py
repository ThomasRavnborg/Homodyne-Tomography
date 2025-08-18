# -*- coding: utf-8 -*-
"""
Created on Fri Aug 15 13:50:36 2025

@author: Thomas Borup Ravnborg
"""

import numpy as np
import pandas as pd
from scipy.special import hermite, factorial
from tqdm import tqdm

def get_overlaps(bin_centers, thetas, N):
    """
    Precompute overlaps <n|x,θ> for all n, bins, and thetas.
    ---------------------------------------------------------
    Inputs:
        bin_centers: quadrature values, ndarray of shape (M,)
        thetas: quadrature phases, ndarray of shape (K,)
        N: Fock space dimension cutoff, int
    ---------------------------------------------------------
    Returns:
        overlaps: dict - keys are (i, j) tuples and values are (N, 1) column vectors
    ---------------------------------------------------------
    """
    overlaps = {}
    for i, theta in enumerate(thetas):
        for j, x in enumerate(bin_centers):
            coeffs = np.zeros(N, dtype=np.complex128)
            # Normalization factor
            norm = (2/np.pi)**(1/4) * np.exp(-x**2)

            for n in range(N):
                Hn = hermite(n)(np.sqrt(2) * x)
                coeffs[n] = norm * Hn / np.sqrt(2**n * factorial(n)) * np.exp(1j * n * theta)
            overlaps[(i, j)] = coeffs[:, np.newaxis]  # column vector
    return overlaps


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

        for i, th in enumerate(thetas):
            for j, bin in enumerate(bin_centers):
                Pi = get_Pi(th, bin, N)
                Pr = np.trace(Pi @ rho).real
                if Pr <= 0:
                    Pr = 1e-15
                R += counts[i, j]/Pr * Pi

        rho = R @ rho @ R
        rho /= np.trace(rho)
    
    return rho

def MaximumLikelihood(thetas, x_values, N=10, num_bins=150, max_iters=200, tol=1e-6):
    bin_centers, counts = bin_X(x_values, num_bins=num_bins)
    overlaps = get_overlaps(bin_centers, thetas, N)
    
    rho = np.eye(N, dtype=np.complex128) / N
    likelihoods = []

    for it in tqdm(range(max_iters)):
        R = np.zeros((N, N), dtype=np.complex128)
        logL = 0.0

        for i, th in enumerate(thetas):
            for j, x in enumerate(bin_centers):
                c = counts[i, j]
                if c == 0:
                    continue
                psi = overlaps[(i, j)]
                Pi = psi @ psi.conj().T
                p = np.trace(Pi @ rho).real
                if p <= 0:
                    p = 1e-15
                R += (c / p) * Pi
                logL += c * np.log(p)

        rho_new = R @ rho @ R
        rho_new /= np.trace(rho_new)

        likelihoods.append(logL)
        rho = rho_new

        # convergence by log-likelihood saturation
        if it > 0 and abs(likelihoods[-1] - likelihoods[-2]) < tol:
            print(f"Converged at iteration {it}, log-likelihood={logL:.6f}")
            break

    return rho, likelihoods




import os
from pathlib import Path
# Loading data
parent = Path(os.path.dirname(os.getcwd()))
date = "091027"
state = "cat2"

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

rho_est, likelihoods = MaximumLikelihood(thetas-theta0, x_values-x0, N=20, num_bins=200, max_iters=400)
np.save("rho_est.npy", rho_est)

