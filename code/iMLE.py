import numpy as np
from scipy.special import hermite, factorial
from tqdm import tqdm


def Pi_mn(theta, x_vals, N):
    """
    Projection operator for multiple x values in Fock basis.

    Inputs:
        theta: Quadrature phase in radians (float)
        x_vals: Array of quadrature values (1D array)
        N: Dimension cutoff

    Returns:
        psi: ndarray of shape (N, len(x_vals))
    """
    x_vals = np.atleast_1d(x_vals)
    num_x = len(x_vals)
    psi = np.zeros((N, num_x), dtype=np.complex128)
    norm = (1 / (np.pi**0.25)) * np.exp(-x_vals**2 / 2)

    for n in range(N):
        Hn = hermite(n)(x_vals)
        psi[n, :] = norm * Hn / np.sqrt(2**n * factorial(n)) * np.exp(-1j * n * theta)

    return psi

def bin_X(quadratures, num_bins=200, range_x=None):
    """
    quadratures: np.array of shape (M, K) — M angles, K measurements per angle
    num_bins: number of histogram bins
    range_x: (min, max) — if None, taken from data
    
    Returns:
        bin_centers: shape (num_bins,)
        counts: shape (M, num_bins) — histogram counts for each angle
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


def iMLE(thetas, x_values, N=10, num_bins=150, max_iters=50, tol=1e-6):
    bin_centers, counts = bin_X(x_values, num_bins=num_bins)
    num_bins = len(bin_centers)
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