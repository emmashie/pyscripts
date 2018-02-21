#!/usr/bin/env python

import numpy as np
import scipy
from numpy.fft import fft, ifft, fftshift, fftfreq
from stompy.utils import model_skill
from stompy.utils import find_lag
from stompy.utils import rms

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
    """ x, y, m (number of subrecords), dt, window='Blackman' or 'None'
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

def model_metrics(tmod, mod, tobs, obs):
    """ calculate model metrics
        mod and obs must be at same time and equivalent length
    """
    valid=np.isfinite(mod+obs)
    mod_val=mod[valid]
    obs_val=obs[valid]
    ms = model_skill(mod_val, obs_val)
    bias = np.mean(mod_val-obs_val)
    r2 = np.corrcoef(mod_val, obs_val)[0,1]**2
    rms_err = rms(mod_val-obs_val)
    lag = find_lag(tmod, mod, tobs, obs)
    # ignoring biases, how closer are we on amplitude?  
    # mb=np.polyfit(mod_val,obs_val,1)
    # amp_factor=mb[0]
    # This way further ignores phase error
    amp_factor=rms(mod_val - mod_val.mean()) / rms(obs_val - obs_val.mean())
    
    return ms, bias, r2, rms_err, lag, amp_factor

