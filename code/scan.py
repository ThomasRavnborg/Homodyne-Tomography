# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 2025

@author: Jan Scarabelli
"""

#%% import necessary libraries

import numpy as np
import json
import os
from natsort import natsorted

from pathlib import Path


#%% define functions
def fs(t, t0):
    """
    Temporal mode function
    """
    f = 9*1e6 # Hz
    gamma = f*2*np.pi
    return np.sqrt(gamma)*np.exp(-gamma*np.abs(t-t0))


def vac_normalise(trace, vac_mean, vac_std):
    """
    Normalise trace wrt vaccum data
    """
    pass
#%% read data
parent = Path.cwd()  # Get the parent directory (repo root directory)
date = "091027"
#state = "cat1"

data_folder = parent / "data" / "processed_data"

states_list = [d.name.split('.')[0] for d in natsorted((data_folder / date).iterdir()) if 
               'dts' not in d.name]

state_ts = {}
for state in states_list:
    dt = json.load(open(data_folder / date / 'dts.json'))[state]
    data = np.load(data_folder / date / (state+'.npy'))

    #%% Get timestamp
    data_angles = np.delete(data, 12, axis=0)  # remove vacuum
    state_vars = data_angles.var(axis=1)  # get variance of 30000 points
    ts = np.argmax(state_vars)  # get timestamp of maximum variance
    state_ts[state] = np.round(ts.mean())  # select avg time