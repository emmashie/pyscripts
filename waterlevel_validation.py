#!/usr/bin/env python

import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt

filename = "/home/emma/validation_data/NOAA_WaterLevel/NOAA_9414290.csv"
hisfile = "/home/emma/sfb_dfm_setup/r14/DFM_OUTPUT_r14/his_files/r14_0000*.nc"

def load_NOAA(filename=filename, sep=',', header=0, usecols=0, time_format='%Y-%m-%dT%H:%M:%SZ'):
	""" loads file (csv) of NOAA WaterLevel Data for a station
    and outputs time (list of datetime objects) and waterlevel 
	
	filename -- string of name and location
	sep -- deliniation between columns (default=',')
	usecols -- choose which columns to load [x,y], first column time, second column zeta 
	time_format -- format of time variable in file (default='%Y-%m-%dT%H:%M:%SZ')
	"""
	dat = pd.read_csv(filename, sep=sep, header=header, usecols=usecols)
	time = []
	for i in range(len(dat.values[:,0])):
		time.append(dt.datetime.strptime(dat.values[i,0], time_format))
	zeta = dat.values[:,1]
	for i in range(len(time)):
		time[i].replace(tzinfo=dt.timezone.utc)
	return time, zeta.astype(float)

def load_model(hisfile=hisfile, rec=0, ref_year=2012, ref_month=8, ref_day=1): 
	""" loads specific observation point record (rec) and 
    converts time variable to datetime objects
	
	hisfile -- string of name and location of history file (netcdf file)
	rec -- the index of the record desired
	ref_year -- reference year for the model output
	ref_month -- reference month for the model output
	ref_day -- reference day for the model output
	"""	
	his = nc.MFDataset(hisfile)
	t_ = his.variables["time"][:]
	waterlevel = his.variables["waterlevel"][:,rec]
	dum = dt.datetime(year=ref_year, month=ref_month, day=ref_day)
	t = []
	for i in range(len(t_)):
		t.append(dt.datetime.fromtimestamp(t_[i]+((dum-dt.datetime.fromtimestamp(t_[0])).days)*24*60*60, tz=dt.timezone.utc))	
	return t, waterlevel
	
def dt2ts(datetimes):
	""" takes a list of datetime objects and converts
    list to array of timestamps
	"""
	time_ts = np.zeros(len(datetimes))
	for i in range(len(datetimes)):
		time_ts[i] = datetimes[i].timestamp()	
	return time_ts
	
