# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 12:34:56 2023

@author: Thomas Ravnborg
"""

# Importing necessary libraries
import numpy as np
from scipy.special import genlaguerre, factorial

# Defining Wmn function
def Wmn(x, p, m, n):
    """
    Wigner function W_mn(x,p) for given m,n and arrays x,p.
    Inputs:
        x, p: 2D numpy arrays from meshgrid.
        m, n: integers
    Returns:
        Wigner function W_mn(x,p) as a 2D complex array
    """
    # Computing r^2 and prefactor
    r2 = x**2 + p**2
    prefactor = np.exp(-r2) / np.pi
    # For m>=n, the Wigner function Wmn is calculated using the equation for Leonhardt
    if m >= n:
        # Computing Laguerre polynomial, phase factor and coefficient
        L = genlaguerre(n, m-n)(2*r2)
        phase_factor = ((-1)**n) * ((x - 1j*p)**(m-n))
        coeff = np.sqrt(2**(m-n) * factorial(n) / factorial(m))
        return prefactor * phase_factor * coeff * L
    # For m<n, the Wigner function Wmn is calculated using the conjugate symmetry
    else:
        return np.conjugate(Wmn(x, p, n, m))

# Defining function to convert from density matrix to Wigner function
def rho_to_Wigner(rho, xgrid, pgrid):
    """
    Convert density matrix to Wigner function (vectorized).
    Inputs:
        rho: density matrix (NxN)
        xgrid, pgrid: 1D arrays defining phase space grid
    Returns:
        Wigner function (2D array)
    """
    # Create meshgrid and 2D array for the Wigner function
    X, P = np.meshgrid(xgrid, pgrid, indexing='ij')
    dim = rho.shape[0]
    # Initialize Wigner function
    W = np.zeros(X.shape, dtype=complex)
    # Calculate Wigner function as sum over density matrix elements with Wmn
    for m in range(dim):
        for n in range(dim):
            W += rho[m, n] * Wmn(X, P, m, n)
    return np.real(W)

# Defining function to convert from Wigner function to density matrix
def Wigner_to_rho(W, xgrid, pgrid, dim, trapz=True):
    """
    Convert Wigner function to density matrix (vectorized).
    Inputs:
        W: Wigner function (2D array)
        xgrid, pgrid: 1D arrays defining phase space grid
        dim: Hilbert space dimension
    Returns:
        Density matrix (2D array)
    """
    # Create meshgrid and calculate dx, dp
    X, P = np.meshgrid(xgrid, pgrid, indexing='ij')
    dx = xgrid[1] - xgrid[0]
    dp = pgrid[1] - pgrid[0]
    # Initialize density matrix
    rho = np.zeros((dim, dim), dtype=complex)
    # Calculate density matrix as integrals over Wigner function with Wmn
    for m in range(dim):
        for n in range(dim):
            integrand = W * Wmn(X, P, n, m)
            if trapz:
                inner = np.trapezoid(integrand, pgrid, axis=1)
                integral = np.trapezoid(inner, xgrid, axis=0)
                rho[m, n] = 2 * np.pi * integral
            else:
                rho[m, n] = 2 * np.pi * np.sum(integrand) * dx * dp
    # Enforce Hermiticity of the density matrix
    rho = 0.5*(rho + rho.conj().T)
    return rho
