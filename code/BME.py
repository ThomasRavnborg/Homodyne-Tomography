# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 2025

@author: Jan Scarabelli
"""

#%% import necessary libraries

import numpy as np
import os
import lecroy
import json

from pathlib import Path
from scipy.fft import rfft, irfft, rfftfreq
from natsort import natsorted
from tqdm import tqdm

#%% Define auxiliary functions

def prior_dist(N, method='HS'):
    
    match method:
        case 'HS':
            G = np.random.normal(size=(N,N)) + 1.j * np.random.normal(size=(N,N)) / np.sqrt(2)
            ggt = G @ G.conjugate().T
            rho = ggt / np.trace(ggt)
        
    return rho


def logL(counts, prob):
    return sum([counts[i]* np.log(prob[i]) for i in range (len(counts))])
    pass


