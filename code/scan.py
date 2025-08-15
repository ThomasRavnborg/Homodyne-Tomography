# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 2025

@author: Jan Scarabelli
"""

#%% import necessary libraries

import numpy as np
import json
import os

from pathlib import Path

#%% read data
parent = Path(os.path.dirname(os.getcwd()))
date = "091027"
state = "cat1"

data_path = parent / "data" / "processed_data" / date / (state + '.npy')

state_dirs = [d.name for d in natsorted((data_folder / date).iterdir()) if d.is_dir() and \
                  ('cat' in d.name or 'tora' in d.name)]
for state in 