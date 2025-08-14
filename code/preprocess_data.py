# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 2025

@author: Jan Scarabelli
"""

#%% import necessary libraries

import numpy as np
import os
import lecroy

from pathlib import Path
from scipy.fft import rfft, irfft, rfftfreq
from natsort import natsorted
from tqdm import tqdm

#%% Define paths, dates and angles
parent = Path.cwd()  # Get the parent directory (repo root directory)
data_folder = parent / 'data' / 'data-tora'  # Path of data folder

dates = os.listdir(data_folder)  # List of dates in the data folder
angles = [f"{i*15:03}" for i in range(12)] + ['vac']


#%% Load and process data for each date and angle

out_dir = parent / 'data' / 'processed_data'  # Output directory for processed data
out_dir.mkdir(parents=True, exist_ok=True)  # Create output directory if it doesn't

# Iterate over files to create a 4D array of data
# Shape: (angles, traces, time values, values per time + angle) = (13, 10000, 127, 3)
for date in dates:
    print(f"Processing date: {date}")
    state_dirs = [d.name for d in natsorted((data_folder / date).iterdir()) if d.is_dir() and \
                  ('cat' in d.name or 'tora' in d.name)]

    for state in tqdm(state_dirs):
        state_path = data_folder / date / state
        data_state = np.empty((13, 10000, 127, 3))  # angles, traces, time values, files per angle
        for i,theta in enumerate(angles):
            files_theta = [f for f in state_path.iterdir() if 'C1cat'+f"{theta}" in f.name or 
                           'C1'+f"{theta}" in f.name]
            data_theta = np.empty((10000, 127, 3))  # Initialize array for data
            for j,f in enumerate(files_theta):
                meta, times, data_theta[:,:,j] = lecroy.read(f, scale=False)
            
            data_state[i,:,:,:] = data_theta  # Store the data in the data_state array

        # create output directory if it doesn't exist
        out_dir_date = out_dir / date
        out_dir_date.mkdir(parents=True, exist_ok=True)
        # save the data for the current state
        np.save(out_dir / date / f'{state}.npy', data_state)
