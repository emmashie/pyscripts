import numpy as np
import matplotlib.pyplot as plt
import pyscripts.waterlevel_validation as wv
from matplotlib.dates import num2date, date2num
import pyscripts.analysis as an
import datetime as dt
from stompy.spatial import proj_utils
import os
import netCDF4 as nc
from stompy.io.local import noaa_coops


#hisfile = "/home/emma/sfb_dfm_setup/r14/DFM_OUTPUT_r14/his_files/r14_0000*.nc"
path = "/opt/data/delft/sfb_dfm_v2/runs/wy2013/DFM_OUTPUT_wy2013/"
hisfile = path + "wy2013_0000_20120801_000000_his.nc"
savepath = path + "/validation_plots/waterlevel_validation_plots/"

if not os.path.exists(savepath):
	os.makedirs(savepath)

ll_to_utm = proj_utils.mapper('WGS84','EPSG:26910')
utm_to_ll = proj_utils.mapper('EPSG:26910','WGS84')

mdat = nc.MFDataset(hisfile)
# pull out coordinates of model stations
xcoor = mdat.variables["station_x_coordinate"][:]
ycoor = mdat.variables["station_y_coordinate"][:]
# convert utm coordinates to lat lon 
mll = np.zeros((2, len(xcoor)))
for i in range(len(xcoor)):
	mll[:,i] = utm_to_ll([xcoor[i], ycoor[i]])


##### station 9414290
dat =  noaa_coops.coops_dataset_product(station="9414290", product="water_level", start_date=np.datetime64("2012-08-01"), end_date=np.datetime64("2013-09-01"), days_per_request=31)
lon = dat["lon"].values[0,0]
lat = dat["lat"].values[0,0]
time = dat["time"]
dtime = [dt.datetime.utcfromtimestamp(time[i].astype(float)/1e9) for i in range(len(time))]
times = date2num(dtime)
waterlevel = dat["water_level"][0,:]
ofreq, ospec = an.band_avg(times, waterlevel, dt=times[1]-times[1])
dist = np.sqrt((mll[0,:]-lon)**2 +  (mll[1,:]-lat)**2)
rec = np.where(dist == np.min(dist))[0]
mtime, mwaterlevel = wv.load_model(hisfile, rec=rec)
mtimes = date2num(mtime)
mwaterleveli = np.interp(times, mtimes, np.asarray(mwaterlevel)[:,0])
mfreq, mspec = an.band_avg(mtimes, np.asarray(mwaterlevel)[:,0])
# plotting
fig, ax = plt.subplots(figsize=(6,4))
ax.plot(time, waterlevel, color='cornflowerblue')
ax.plot(mtime, mwaterlevel, color='turquoise', alpha=0.75)
ax.set_title("San Francisco : 9414290")
ax.legend(["obs","model"], loc='best')
ax.set_ylabel("m")
ax.set_xlim([dt.date(2012,10,1), dt.date(2012,10,7)])
fig.autofmt_xdate()
fig.savefig(savepath + "SanFrancisco.png")


##### station 9415020
dat =  noaa_coops.coops_dataset_product(station="9415020", product="water_level", start_date=np.datetime64("2012-08-01"), end_date=np.datetime64("2013-09-01"), days_per_request=31)
lon = dat["lon"].values[0,0]
lat = dat["lat"].values[0,0]
time = dat["time"]
dtime = [dt.datetime.utcfromtimestamp(time[i].astype(float)/1e9) for i in range(len(time))]
times = date2num(dtime)
waterlevel = dat["water_level"][0,:]
ofreq, ospec = an.band_avg(times, waterlevel, dt=times[1]-times[1])
dist = np.sqrt((mll[0,:]-lon)**2 +  (mll[1,:]-lat)**2)
rec = np.where(dist == np.min(dist))[0]
mtime, mwaterlevel = wv.load_model(hisfile, rec=rec)
mtimes = date2num(mtime)
mwaterleveli = np.interp(times, mtimes, np.asarray(mwaterlevel)[:,0])
mfreq, mspec = an.band_avg(mtimes, np.asarray(mwaterlevel)[:,0])
# plotting
fig, ax = plt.subplots(figsize=(6,4))
ax.plot(time, waterlevel, color='cornflowerblue')
ax.plot(mtime, mwaterlevel, color='turquoise', alpha=0.75)
ax.set_title("Point Reyes : 9415020")
ax.legend(["obs","model"], loc='best')
ax.set_ylabel("m")
ax.set_xlim([dt.date(2012,10,1), dt.date(2012,10,7)])
fig.autofmt_xdate()
fig.savefig(savepath + "PointReyes.png")



##### station 9414863
dat =  noaa_coops.coops_dataset_product(station="9414863", product="water_level", start_date=np.datetime64("2012-08-01"), end_date=np.datetime64("2013-09-01"), days_per_request=31)
lon = dat["lon"].values[0,0]
lat = dat["lat"].values[0,0]
time = dat["time"]
dtime = [dt.datetime.utcfromtimestamp(time[i].astype(float)/1e9) for i in range(len(time))]
times = date2num(dtime)
waterlevel = dat["water_level"][0,:]
ofreq, ospec = an.band_avg(times, waterlevel, dt=times[1]-times[1])
dist = np.sqrt((mll[0,:]-lon)**2 +  (mll[1,:]-lat)**2)
rec = np.where(dist == np.min(dist))[0]
mtime, mwaterlevel = wv.load_model(hisfile, rec=rec)
mtimes = date2num(mtime)
mwaterleveli = np.interp(times, mtimes, np.asarray(mwaterlevel)[:,0])
mfreq, mspec = an.band_avg(mtimes, np.asarray(mwaterlevel)[:,0])
# plotting
fig, ax = plt.subplots(figsize=(6,4))
ax.plot(time, waterlevel, color='cornflowerblue')
ax.plot(mtime, mwaterlevel, color='turquoise', alpha=0.75)
ax.set_title("Richmond : 9414863")
ax.legend(["obs","model"], loc='best')
ax.set_ylabel("m")
ax.set_xlim([dt.date(2012,10,1), dt.date(2012,10,7)])
fig.autofmt_xdate()
fig.savefig(savepath + "Richmond.png")



##### station 9414750
dat =  noaa_coops.coops_dataset_product(station="9414750", product="water_level", start_date=np.datetime64("2012-08-01"), end_date=np.datetime64("2013-09-01"), days_per_request=31)
lon = dat["lon"].values[0,0]
lat = dat["lat"].values[0,0]
time = dat["time"]
dtime = [dt.datetime.utcfromtimestamp(time[i].astype(float)/1e9) for i in range(len(time))]
times = date2num(dtime)
waterlevel = dat["water_level"][0,:]
ofreq, ospec = an.band_avg(times, waterlevel, dt=times[1]-times[1])
dist = np.sqrt((mll[0,:]-lon)**2 +  (mll[1,:]-lat)**2)
rec = np.where(dist == np.min(dist))[0]
mtime, mwaterlevel = wv.load_model(hisfile, rec=rec)
mtimes = date2num(mtime)
mwaterleveli = np.interp(times, mtimes, np.asarray(mwaterlevel)[:,0])
mfreq, mspec = an.band_avg(mtimes, np.asarray(mwaterlevel)[:,0])
# plotting
fig, ax = plt.subplots(figsize=(6,4))
ax.plot(time, waterlevel, color='cornflowerblue')
ax.plot(mtime, mwaterlevel, color='turquoise', alpha=0.75)
ax.set_title("Alameda : 9414750")
ax.legend(["obs","model"], loc='best')
ax.set_ylabel("m")
ax.set_xlim([dt.date(2012,10,1), dt.date(2012,10,7)])
fig.autofmt_xdate()
fig.savefig(savepath + "Alameda.png")



##### station 9414523
dat =  noaa_coops.coops_dataset_product(station="9414523", product="water_level", start_date=np.datetime64("2012-08-01"), end_date=np.datetime64("2013-09-01"), days_per_request=31)
lon = dat["lon"].values[0,0]
lat = dat["lat"].values[0,0]
time = dat["time"]
dtime = [dt.datetime.utcfromtimestamp(time[i].astype(float)/1e9) for i in range(len(time))]
times = date2num(dtime)
waterlevel = dat["water_level"][0,:]
ofreq, ospec = an.band_avg(times, waterlevel, dt=times[1]-times[1])
dist = np.sqrt((mll[0,:]-lon)**2 +  (mll[1,:]-lat)**2)
rec = np.where(dist == np.min(dist))[0]
mtime, mwaterlevel = wv.load_model(hisfile, rec=rec)
mtimes = date2num(mtime)
mwaterleveli = np.interp(times, mtimes, np.asarray(mwaterlevel)[:,0])
mfreq, mspec = an.band_avg(mtimes, np.asarray(mwaterlevel)[:,0])
# plotting
fig, ax = plt.subplots(figsize=(6,4))
ax.plot(time, waterlevel, color='cornflowerblue')
ax.plot(mtime, mwaterlevel, color='turquoise', alpha=0.75)
ax.set_title("Redwood City : 9414523")
ax.legend(["obs","model"], loc='best')
ax.set_xlim([dt.date(2012,10,1), dt.date(2012,10,7)])
fig.autofmt_xdate()
fig.savefig(savepath + "RedwoodCity.png")



#fig, ax = plt.subplots(nrows=4, sharex=True, figsize=(10,10))
#ax[0].plot(time_1, zeta_1-np.mean(zeta_1), color='cornflowerblue')
#ax[0].plot(t_1, waterlevel_1, color='turquoise', alpha=0.75)
#ax[0].plot(time_1, waterlevel_i_1 - (zeta_1-np.mean(zeta_1)), color='lightcoral')
#ax[0].set_title("9414290")
#ax[0].legend(["obs","model", "model-obs"], loc='best')

#ax[1].plot(time_2, zeta_2-np.mean(zeta_2), color='cornflowerblue')
#ax[1].plot(t_2, waterlevel_2, color='turquoise', alpha=0.75)
#ax[1].plot(time_2, waterlevel_i_2 - (zeta_2-np.mean(zeta_2)), color='lightcoral')
#ax[1].set_title("9415020")

#ax[2].plot(time_3, zeta_3-np.mean(zeta_3), color='cornflowerblue')
#ax[2].plot(t_3, waterlevel_3, color='turquoise', alpha=0.75)
#ax[2].plot(time_3, waterlevel_i_3 - (zeta_3-np.mean(zeta_3)), color='lightcoral')
#ax[2].set_title("9414863")

#ax[3].plot(time_4, zeta_4-np.mean(zeta_4), color='cornflowerblue')
#ax[3].plot(t_4, waterlevel_4, color='turquoise', alpha=0.75)
#ax[3].plot(time_4, waterlevel_i_4 - (zeta_4-np.mean(zeta_4)), color='lightcoral')
#ax[3].set_title("9414750")
#fig.savefig(savepath + "waterlevel_timeseries.png")

#fig, ax = plt.subplots(nrows=4, sharex=True, figsize=(10,10))
#ax[0].loglog(ofreq_1, ospec_1, color='cornflowerblue')
#ax[0].loglog(mfreq_1, mspec_1, color='turquoise')
#ax[0].legend(["obs","model"], loc='best')
#ax[0].set_title("9414290")

#ax[1].loglog(ofreq_2, ospec_2, color='cornflowerblue')
#ax[1].loglog(mfreq_2, mspec_2, color='turquoise')
#ax[1].set_ylabel("spectral energy [$m^2 / cph$]")
#ax[1].set_title("9415020")

#ax[2].loglog(ofreq_3, ospec_3, color='cornflowerblue')
#ax[2].loglog(mfreq_3, mspec_3, color='turquoise')
#ax[2].set_title("9414863")

#ax[3].loglog(ofreq_4, ospec_4, color='cornflowerblue')
#ax[3].loglog(mfreq_4, mspec_4, color='turquoise')
#ax[3].set_xlabel("frequency [$cph$]")
#ax[3].set_title("9414750")
#ax[3].set_xlim((10**-4, 0.5))
#fig.savefig(savepath + "waterlevel_spectra.png")

