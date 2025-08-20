# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 2025

@author: Jan Scarabelli
"""

#%% import necessary libraries

import numpy as np
import pandas as pd
import json
import os
from natsort import natsorted
from tqdm import tqdm

from pathlib import Path
from utils import fs, calculate_quadratures

#%% define data paths
parent = Path.cwd()  # Get the parent directory (repo root directory)

data_folder = parent / "data" / "processed_data"
out_dir = parent / "data" / "dataframes"

dates = ["091027","091028","091029"]
angles = [f"{i*15:03}" for i in range(12)]

#%% start loop for all dates and states
for date in dates:
    print(f"Processing date {date}")

    out_dir_date = out_dir / date
    out_dir_date.mkdir(parents=True, exist_ok=True)

    states_list = [d.name.split('.')[0] for d in natsorted((data_folder / date).iterdir()) if 
               'dts' not in d.name]
    
    for state in tqdm(states_list, ascii="░▒▓"):
        dt = json.load(open(data_folder / date / 'dts.json'))[state]
        data_state = np.load(data_folder / date / (state+'.npy'))

        # Get timestamps
        data_angles = np.delete(data_state, 12, axis=0)  # remove vacuum
        state_vars = data_angles.var(axis=1)  # get variance of 30000 points
        timestamps = np.argmax(state_vars, axis=1)  # get timestamp of maximum variance
        t0 = int(np.round(timestamps.mean()))

        # Scanning quadratures
        x_theta = calculate_quadratures(data=data_state, dt=dt, t0=t0)
        df_state = pd.DataFrame(x_theta.T, columns=angles)
        df_state.to_csv((out_dir_date / f'{state}.csv'), index=False)

