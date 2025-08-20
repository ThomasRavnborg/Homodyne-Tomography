import numpy as np
from scipy.special import hermite, factorial

def fs(t, t0):
    f = 9*1e6 # Hz
    gamma = 2*np.pi*f
    return np.sqrt(gamma)*np.exp(-gamma*np.abs(t-t0))

def calculate_quadratures(data, dt, t0=39):

    N = data.shape[2]
    t = np.linspace(0, dt*N, N, endpoint=False)

    # Define temporal mode from peak in variance
    temporal_mode = fs(t, t[t0])  # shape (T,)

    # Extract vacuum data
    vacuum = data[-1,:,:]

    # Calculating vacuum quadratures
    vacuum_quadratures = vacuum @ temporal_mode * dt
    vacuum_mean = np.mean(vacuum_quadratures)
    vacuum_std = np.std(vacuum_quadratures)

    x_values = np.empty((12, 30000))

    for i in range(12):
        state = data[i,:,:]
        # Project each trace onto the temporal mode
        quadratures = state @ temporal_mode * dt
        quadratures = (quadratures - vacuum_mean) /(np.sqrt(2)*vacuum_std)
        x_values[i] = quadratures

    return x_values


def get_overlaps(theta, x_vals, N):
    """
    Precompute overlaps psi = <n|x,θ> for all n and quadrature values.
    
    Inputs:
        theta : Quadrature phase in radians (float)
        x_vals : Quadrature values (ndarray, shape (M, K))
        N : Fock space cutoff dimension (int)
    
    Returns:
        psi : overlap vector |x,θ> (ndarray, shape (N, M, K))
    """
    # Ensure x_vals is a 1D array and count number of elements
    x_vals = np.atleast_1d(x_vals)
    num_x = len(x_vals)

    # Initialize overlap vector and calculate norms
    psi = np.zeros((N, num_x), dtype=np.complex128)
    norms = (1 / (np.pi**0.25)) * np.exp(-x_vals**2 / 2)

    # Calculate overlap vector for each Fock space index
    for n in range(N):
        Hn = hermite(n)(x_vals)
        psi[n, :] = norms * Hn / np.sqrt(2**n * factorial(n)) * np.exp(-1j * n * theta)
    
    return psi

def bin_X(x_vals, num_bins=200, range_x=None):
    """
    Bin quadrature values into histograms.
    
    Inputs:
        x_vals : Quadrature values (ndarray, shape (M, K))
        num_bins : Number of histogram bins (int)
        range_x : tuple (min, max) or None
            If None, taken from global min/max of data
    
    Returns:
        bin_centers : ndarray, shape (num_bins,)
        counts : ndarray, shape (M, num_bins)
    """
    M, K = x_vals.shape
    if range_x is None:
        min_x, max_x = np.min(x_vals), np.max(x_vals)
    else:
        min_x, max_x = range_x

    counts = np.zeros((M, num_bins), dtype=int)
    bin_edges = np.linspace(min_x, max_x, num_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    for i in range(M):
        counts[i], _ = np.histogram(x_vals[i], bins=bin_edges)

    return bin_centers, counts


def log_likelihood(rho, psi_all, counts, M, dx=1):
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


def log_prior(rho):
    """
    Log-prior for density matrices.
    Simple choice: uniform over positive semidefinite, trace-1 matrices.
    Returns -inf if rho is not valid (reject in MH step).
    """
    # Check Hermitian
    if not np.allclose(rho, rho.conj().T):
        return -np.inf
    # Check positive semidefinite
    if np.any(np.linalg.eigvalsh(rho) < 0):
        return -np.inf
    # Check trace 1
    if not np.isclose(np.trace(rho), 1.0):
        return -np.inf
    return 0.0  # uniform prior

def accept_rho(rho, rho_new, psi_all, counts, M, logL_old):
    """
    Metropolis-Hastings acceptance with prior.
    """
    logL_new = log_likelihood(rho_new, psi_all, counts, M)
    log_prior_old = log_prior(rho)
    log_prior_new = log_prior(rho_new)

    logA = (logL_new + log_prior_new) - (logL_old + log_prior_old)
    
    if np.log(np.random.rand()) < logA:
        return True, logL_new
    else:
        return False, logL_old