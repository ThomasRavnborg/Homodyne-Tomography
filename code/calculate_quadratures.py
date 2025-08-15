import numpy as np
import os
from pathlib import Path
import json

def fs(t, t0):
    f = 9*1e6 # Hz
    gamma = 2*np.pi*f
    return np.sqrt(gamma)*np.exp(-gamma*np.abs(t-t0))

# Loading data
parent = Path(os.path.dirname(os.getcwd()))
date = "091027"
state = "cat2"
data_path = parent / "data" / "processed_data" / date
data = np.load(data_path / (state + '.npy'))

dt = json.load(open(data_path / 'dts.json'))[state]

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


    x_values = np.array([])
    theta_values = np.array([])

    for i in range(12):
        state = data[i,0,:,:]
        # Project each trace onto the temporal mode
        quadratures = state @ temporal_mode * dt
        # Normalize quadratures to vacuum
        quadratures = (quadratures - vacuum_mean) /(np.sqrt(2)*vacuum_std)
        # Append quadrature values and angles
        x_values = np.append(x_values, quadratures)
        theta_values = np.append(theta_values, np.full_like(quadratures, 15*i))
        theta_values = np.deg2rad(theta_values)