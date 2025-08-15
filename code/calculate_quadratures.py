import numpy as np

def fs(t, t0):
    f = 9*1e6 # Hz
    gamma = 2*np.pi*f
    return np.sqrt(gamma)*np.exp(-gamma*np.abs(t-t0))

def calculate_quadratures(data, dt):

    N = data.shape[3]
    t = np.linspace(0, dt*N, N, endpoint=False)

    # Define temporal mode from peak in variance
    temporal_mode = fs(t, t[39])  # shape (T,)

    # Extract vacuum data
    vacuum = data[-1,0,:,:]

    # Calculating vacuum quadratures
    vacuum_quadratures = vacuum @ temporal_mode * dt
    vacuum_mean = np.mean(vacuum_quadratures)
    vacuum_std = np.std(vacuum_quadratures)

    x_values = np.empty((12, 10000))

    for i in range(12):
        state = data[i,0,:,:]
        # Project each trace onto the temporal mode
        quadratures = state @ temporal_mode * dt
        quadratures = (quadratures - vacuum_mean) /(np.sqrt(2)*vacuum_std)
        x_values[i] = quadratures