import numpy as np
import netCDF4 as nc
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib.dates import date2num
from stompy.utils import model_skill
import pyscripts.analysis as an

model_files = "/home/emma/sfb_dfm_setup/r14/DFM_OUTPUT_r14/his_files/r14_0000_201*.nc"
adcp_file = "/opt/data/noaa/ports/SFB1312-2013.nc"

# load in model and adcp data as netcdf databases 
mdat = nc.MFDataset(model_files)
dat = nc.Dataset(adcp_file)
ind = 150 # start at 115 for SFB1201-2012, start at 140 for SFB1301-2013

# define variables 
mtime = mdat.variables["time"][:]
mu = mdat.variables["x_velocity"][:]
mv = mdat.variables["y_velocity"][:]
mubar = np.mean(mu, axis=-1)
mvbar = np.mean(mv, axis=-1)
mdep = mdat.variables["zcoordinate_c"][:]
# take model time (referenced to 2012-08-01) and create datetime objects to use for plotting
mdtime = []
for i in range(len(mtime)):
	mdtime.append(dt.datetime.fromtimestamp(mtime[i] + 15553*86400, tz=dt.timezone.utc))
# convert datetime objects to datenums to use for interpolation 
mtimes = date2num(mdtime)

# define variables
time = dat.variables["time"][:]
u = dat.variables["u"][:]
v = dat.variables["v"][:]
ubar = dat.variables["u_davg"][:]
vbar = dat.variables["v_davg"][:]
dep = -dat.variables["depth"][:]
# take observation time and create datetime objects 
dtime = []
for i in range(len(time)):
	dtime.append(dt.datetime.fromtimestamp(time[i], tz=dt.timezone.utc))
# convert datetime objects to datenums to use for interpolation 
dtimes = date2num(dtime)

# interpolate model to observation depths
mu_i = np.zeros((len(u[:,0]), len(mu[:,ind,0])))
mv_i = np.zeros((len(u[:,0]), len(mv[:,ind,0])))
for i in range(len(mdtime)):
	mu_i[:,i] = np.interp(dep, mdep[i,ind,:], mu[i,ind,:])
	mv_i[:,i] = np.interp(dep, mdep[i,ind,:], mu[i,ind,:])
	
# interpolate model to observation times
mubar_i = np.interp(dtimes, mtimes, mubar[:,ind])
mvbar_i = np.interp(dtimes, mtimes, mvbar[:,ind])
	
# computing spectra
mufreq, muspec = an.band_avg(mtimes, mubar[:,ind])
mvfreq, mvspec = an.band_avg(mtimes, mvbar[:,ind])

#n = 11700
#n = 3603
n = 0
oufreq, ouspec = an.band_avg(time[n:], ubar[n:], dt=6/60)
ovfreq, ovspec = an.band_avg(time[n:], vbar[n:], dt=6/60) 

	
# plotting up model & observation time series	
fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(12,6))
ax[0].plot(mdtime[:], mubar[:,ind], color='lightcoral')
ax[0].plot(dtime[:], ubar[:], color='turquoise')
ax[0].legend(["model","adcp"])
ax[1].plot(mdtime[:], mvbar[:,ind], color='lightcoral')
ax[1].plot(dtime[:], vbar[:], color='turquoise')
ax[1].set_xlim((dtime[0], dtime[-1]))

#n = 11700
#n = 3603
np.min(mubar_i[n:] - ubar[n:])
np.max(mubar_i[n:] - ubar[n:])
np.mean(mubar_i[n:] - ubar[n:])
model_skill(mubar_i[n:], ubar[n:])
np.corrcoef(mubar_i[n:], ubar[n:])**2	

np.min(mvbar_i[n:] - vbar[n:])
np.max(mvbar_i[n:] - vbar[n:])
np.mean(mvbar_i[n:] - vbar[n:])
model_skill(mvbar_i[n:], vbar[n:])
np.corrcoef(mvbar_i[n:], vbar[n:])**2	

fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(8,12))
ax[0].loglog(mufreq, muspec, color='lightcoral')
ax[0].loglog(oufreq, ouspec, 'turquoise')
ax[0].legend(["model", "adcp"], loc='best')
ax[0].set_ylabel("spectral energy [$(m/s)^2/cph$]")

ax[1].loglog(mvfreq, mvspec, color='lightcoral')
ax[1].loglog(ovfreq, ovspec, 'turquoise')
ax[1].set_xlabel("frequency [$cph$]")	
ax[1].set_xlim((10**-4, 0.5))
	

plt.subplots()
plt.scatter(mubar[:,ind], mvbar[:,ind], color='lightcoral')
plt.scatter(ubar, vbar, color='turquoise')
plt.legend(["model", "adcp"])
plt.xlabel("u")
plt.ylabel("v")

	

