#!/usr/bin/env python

import numpy as np
import scipy
from numpy.fft import fft, ifft, fftshift, fftfreq
from stompy.utils import model_skill

def blackman_window(width=0):
    x = np.linspace(-1, 1, width + 2, endpoint=True) * np.pi
    x = x[1:-1]
    w = 0.42 + 0.5 * np.cos(x) + 0.08 * np.cos(2 * x)
    weights = w/w.sum()
    return weights

def detrend(x=0, y=0):
    slope, intercept = np.polyfit(x, y, 1)
    y_lin = intercept + slope*x
    y_new = y - y_lin 
    return y_new

def band_avg(x=0, y=0, m=5, dt=1, window='Blackman'):
    """ x, y, m (number of subrecords), window='Blackman' or 'None'
    """
    N = len(x)
    y_new = detrend(x, y)   
    if window == 'Blackman':
        weights = blackman_window(N)
        y_new = y_new * weights
    elif window == 'None':
        y_new = y_new
    else:
        raise ValueError("Invalid option. 'Blackman' or 'None' allowed")
        return    
    fy = fftshift(fft(y_new))
    freqs = fftshift(fftfreq(len(x), dt))   
    spectra = (abs(fy)**2 * dt)/ len(x)    
    boxcar = np.ones(m)/m    
    spectra_avg = np.convolve(boxcar, spectra, mode='same')       
    return freqs, spectra_avg

def model_metrics(mod, obs):
	ms = model_skill(mod, obs)
	bias = np.mean(mod, obs)
	r2 = np.corrcoef(mod, obs)[0,1]**2
	rms = np.mean(np.sqrt(mod - obs)**2))
	return ms, bias, r2, rms
