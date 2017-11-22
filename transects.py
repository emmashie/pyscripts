import xarray as xr
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import cmocean.cm as cmo
import datetime as dt
import pandas as pd
import os

from stompy import utils
from stompy.plot import plot_utils
from stompy.io.local import usgs_sfbay
from stompy.spatial import proj_utils

ll_to_utm = proj_utils.mapper('WGS84','EPSG:26910')
utm_to_ll = proj_utils.mapper('EPSG:26910','WGS84')

### Load in USGS cruise data, define variables, and convert datetime64 times to timestamps
ds=usgs_sfbay.cruise_dataset(np.datetime64('2012-08-01'), np.datetime64('2013-10-01'))
lat = np.asarray(ds["latitude"])
lon = np.asarray(ds["longitude"])
salt = np.asarray(ds["salinity"])
temp = np.asarray(ds["temperature"])
depth = np.asarray(ds["depth"])
dist = np.asarray(ds["Distance_from_station_36"])
time = np.asarray(ds["time"])
times = (time - np.datetime64('1969-12-31T17:00:00Z')) / np.timedelta64(1, 's')

### fix nans in time record and fill in with valid times
tvalid = utils.fill_invalid(times, axis=-1)
tvalid = utils.fill_invalid(tvalid, axis=1)
tvalid = utils.fill_invalid(tvalid, axis=0)
times = tvalid

### Load in his files, define variables, adjust times to same reference time as USGS data
path = "/opt/data/delft/sfb_dfm_v2/runs/wy2013a/DFM_OUTPUT_wy2013a/"
hisfile = "wy2013a_0000_20120801_000000_his.nc"
#his = nc.MFDataset("/home/emma/sfb_dfm_setup/r14/DFM_OUTPUT_r14/his_files/r14_0000*.nc")

temp = False

his = nc.MFDataset(path + hisfile)
xcoor = his.variables["station_x_coordinate"][:]
ycoor = his.variables["station_y_coordinate"][:]
dref = 15553*24*60*60
dtz = 7*60*60
mtimes = his.variables["time"][:]+dref+dtz
msalt = his.variables["salinity"][:]
if temp == True:
	mtemp = his.variables["temperature"][:]
mdepth = his.variables["zcoordinate_c"][:]

mll = np.zeros((2, len(xcoor)))
for i in range(len(xcoor)):
	mll[:,i] = utm_to_ll([xcoor[i], ycoor[i]])

llind = np.zeros(len(ds["latitude"]))
for i in range(len(llind)):
	dist_ = np.sqrt((mll[0,:]-lon[i])**2 +  (mll[1,:]-lat[i])**2)
	llind[i] = np.where(dist_ == np.min(dist_))[0]

tind = np.zeros(times[:,:,0].shape)
for i in range(len(tind[:,0])):
	for j in range(len(tind[0,:])):
		tind[i,j] = utils.nearest(mtimes, times[i,j,0])

mdates = np.zeros((len(tind[:,0]), len(tind[0,:]), len(mdepth[0,0,:])))
msalt_ = np.zeros((len(tind[:,0]), len(tind[0,:]), len(mdepth[0,0,:])))
if temp == True:
	mtemp_ = np.zeros((len(tind[:,0]), len(tind[0,:]), len(mdepth[0,0,:])))
mdepth_ = np.zeros((len(tind[:,0]), len(tind[0,:]), len(mdepth[0,0,:])))
for i in range(len(msalt_[:,0,0])):
	for j in range(len(msalt_[0,:,0])):
		msalt_[i,j,:] = msalt[tind[i,j], llind[j], :]
		mdepth_[i,j,:] = mdepth[tind[i,j], llind[j], :]
		mdates[i,j,:] = mtimes[tind[i,j]]
		if temp == True:
			mtemp_[i,j,:] = mtemp[tind[i,j], llind[j], :]


msalt_ = np.ma.masked_where(msalt_ == -999., msalt_)
if temp == True:
	mtemp_ = np.ma.masked_where(mtemp_ == -999., mtemp_)
mdepth_ = np.ma.masked_where(mdepth_ == -999., mdepth_)

mdist_ = np.zeros((len(dist), len(mdepth_[0,0,:])))
for i in range(len(mdepth_[0,0,:])):
	mdist_[:,i] = dist[:]

if temp == True:
	mdat = xr.Dataset({'salinity': (['Distance_from_station_36', 'prof_sample'], msalt_[0,:,:]),
				   'temperature': (['Distance_from_station_36', 'prof_sample'], mtemp_[0,:,:])}, 
				 coords={'Distance_from_station_36': (['Distance_from_station_36'], dist), 
						 'prof_sample': (['prof_sample'], np.arange(10)),
						 'depth': (['Distance_from_station_36', 'prof_sample'], mdepth_[0,:,:])})						 
	mdat = xr.Dataset({'salinity': (['date', 'Distance_from_station_36', 'prof_sample'], msalt_),
				   'temperature': (['date', 'Distance_from_station_36', 'prof_sample'], mtemp_)}, 
				 coords={'date': (['date'], np.arange(len(tind))),
						 'Distance_from_station_36': (['Distance_from_station_36'], dist), 
						 'prof_sample': (['prof_sample'], np.arange(10)),
						 'depth': (['date', 'Distance_from_station_36', 'prof_sample'], mdepth_),
						 'times': (['date', 'Distance_from_station_36', 'prof_sample'], mdates)})
if temp == False:
	mdat = xr.Dataset({'salinity': (['Distance_from_station_36', 'prof_sample'], msalt_[0,:,:])},
				 coords={'Distance_from_station_36': (['Distance_from_station_36'], dist), 
						 'prof_sample': (['prof_sample'], np.arange(10)),
						 'depth': (['Distance_from_station_36', 'prof_sample'], mdepth_[0,:,:])})						 
	mdat = xr.Dataset({'salinity': (['date', 'Distance_from_station_36', 'prof_sample'], msalt_)},
				 coords={'date': (['date'], np.arange(len(tind))),
						 'Distance_from_station_36': (['Distance_from_station_36'], dist), 
						 'prof_sample': (['prof_sample'], np.arange(10)),
						 'depth': (['date', 'Distance_from_station_36', 'prof_sample'], mdepth_),
						 'times': (['date', 'Distance_from_station_36', 'prof_sample'], mdates)})

st_names = ["Calaveras Point", "Mowry Slough", "Newark Slough", "Dumbarton Bridge", "Ravenswood Point", 
			"Coyote Hills", "Redwood Creek", "Steinberger Slough", "S. San Mateo Bridge", "N. San Mateo Bridge",
			"SF Airport", "San Bruno Shoal", "Oyster Point", "Candlestick Point", "Hunter's Point", "Potrero Point",
			"Bay Bridge", "Blossom Rock", "Point Blunt", "Racoon Strait", "Charlie Buoy", "Point San Pablo",
			"Echo Buoy", "N. Pinole Point", "Pinole Shoal", "Mare Island", "Corckett", "Benicia", "Martinez",
			"Avon Pier", "Roe Island", "Middle Ground", "Simmons Point", "Pittsburg", "Chain Island", "Sacramento River",
			"Rio Vista"]

# set up figures directory
figpath = path + "validation_plots/usgs_transects/"
if not os.path.exists(figpath):
	os.makedirs(figpath)

# avg times for plotting and figure names
t = np.nanmean(np.nanmean(times,axis=-1),axis=-1)
for d in range(len(tind[:,0])):
	fig,ax=plt.subplots(nrows=3, figsize=(8.5,9.5), sharex=True)
	ax[0].axis('tight')
	cruise=ds.isel(date=d) # choose a single cruise
	field='salinity'
	obs=plot_utils.transect_tricontourf(cruise[field],ax=ax[0],V=np.linspace(0,35,36),
                                     cmap=cmo.matter_r,
                                     xcoord='Distance_from_station_36',
                                     ycoord='depth',
									 extend='both')
	ax[0].set_ylabel('Depth (m)')
	#cbar0 = plt.colorbar(obs,label=field, ax=ax[0])
	cbar0 = plt.colorbar(obs,label=field, cax=plt.gcf().add_axes((0.93,0.645,0.01,0.24)))
	ax[0].set_title('cruise# %s' % dt.datetime.fromtimestamp(t[d]).strftime('%Y%m%d'))
	ax[1].axis('tight')
	mod=plot_utils.transect_tricontourf(mdat.isel(date=d)[field],ax=ax[1],V=np.linspace(0,35,36),
                                     cmap=cmo.matter_r,
                                     xcoord='Distance_from_station_36',
                                     ycoord='depth',
									 extend='both')
	ax[1].set_ylabel('Depth (m)')
	#cbar1 = plt.colorbar(mod,label=field, ax=ax[1])
	cbar1 = plt.colorbar(mod,label=field, cax=plt.gcf().add_axes((0.93,0.375,0.01,0.24)))
	ln1 = ax[2].plot(mdat.isel(date=d)["Distance_from_station_36"].values, np.nanmean(mdat.isel(date=d)[field].values, axis=-1), 'o-', label="Model - Depth Avg", color="lightseagreen", markersize=4)
	ln2 = ax[2].plot(cruise["Distance_from_station_36"].values, np.nanmean(cruise[field].values, axis=-1), '^--', label="Obs - Depth Avg", color="lightseagreen", markersize=4)
	ax[2].tick_params('y', colors="lightseagreen")
	ax[2].set_ylabel("Salt [ppt]", color="lightseagreen")
	ax2 = ax[2].twinx()
	ln3 = ax2.plot(mdat.isel(date=d)["Distance_from_station_36"].values, np.abs(np.nanmin(mdat.isel(date=d)[field],axis=-1)-np.nanmax(mdat.isel(date=d)[field],axis=-1)), 'o-', label="Model - Stratificaiton", color="slateblue", alpha=0.8, markersize=4)
	ln4 = ax2.plot(cruise["Distance_from_station_36"].values, np.abs(np.nanmin(cruise[field].values,axis=-1) - np.nanmax(cruise[field].values,axis=-1)), '^--', label="Obs - Stratificaiton", color="slateblue", alpha=0.8, markersize=4)
	ax2.tick_params('y', colors="slateblue")
	ax2.set_ylabel("Stratification (ds/dz) [ppt]", color="slateblue")
	lns = ln1+ln2+ln3+ln4
	labels = [l.get_label() for l in lns]
	ax[2].legend(lns, labels, loc='best', prop={'size': 9})
	ax[2].set_xticks(cruise["Distance_from_station_36"].values[::3])
	ax[2].set_xticklabels(st_names[::3], rotation=45)
	fig.savefig(figpath + "cruise_" + dt.datetime.fromtimestamp(t[d]).strftime('%Y%m%d') + "_salt.png")
	
	if temp == True:
		fig,ax=plt.subplots(nrows=3, figsize=(8.5,7), sharex=True)
		ax[0].axis('tight')
		cruise=ds.isel(date=d) # choose a single cruise
		field='temperature'
		obs=plot_utils.transect_tricontourf(cruise[field],ax=ax[0],V=np.linspace(14,25,36),
                                     cmap='viridis',
                                     xcoord='Distance_from_station_36',
                                     ycoord='depth',
									 extend='both')
		ax[0].set_ylabel('Depth (m)')
		#cbar0 = plt.colorbar(obs,label=field, ax=ax[0])
		cbar0 = plt.colorbar(obs,label=field, cax=plt.gcf().add_axes((0.93,0.645,0.01,0.24)))
		ax[0].set_title('cruise# %s' % dt.datetime.fromtimestamp(t[d]).strftime('%Y%m%d'))
		ax[1].axis('tight')
		mod=plot_utils.transect_tricontourf(mdat.isel(date=d)[field],ax=ax[1],V=np.linspace(14,25,36),
                                     cmap='viridis',
                                     xcoord='Distance_from_station_36',
                                     ycoord='depth',
									 extend='both')
		ax[1].set_ylabel('Depth (m)')
		#cbar1 = plt.colorbar(mod,label=field, ax=ax[1])
		cbar1 = plt.colorbar(mod,label=field, cax=plt.gcf().add_axes((0.93,0.375,0.01,0.24)))
		ln1 = ax[2].plot(mdat.isel(date=d)["Distance_from_station_36"].values, np.nanmean(mdat.isel(date=d)[field].values, axis=-1), 'o-', label="Model - Depth Avg", color="lightseagreen", markersize=4)
		ln2 = ax[2].plot(cruise["Distance_from_station_36"].values, np.nanmean(cruise[field].values, axis=-1), '^--', label="Obs - Depth Avg", color="lightseagreen", markersize=4)
		ax[2].tick_params('y', colors="lightseagreen")
		ax2 = ax[2].twinx()
		ln3 = ax2.plot(mdat.isel(date=d)["Distance_from_station_36"].values, np.abs(np.nanmin(mdat.isel(date=d)[field],axis=-1)-np.nanmax(mdat.isel(date=d)["salinity"],axis=-1)), 'o-', label="Model - Stratificaiton", color="slateblue", alpha=0.8, markersize=4)
		ln4 = ax2.plot(cruise["Distance_from_station_36"].values, np.abs(np.nanmin(cruise[field].values,axis=-1) - np.nanmax(cruise[field].values,axis=-1)), '^--', label="Obs - Stratificaiton", color="slateblue", alpha=0.8, markersize=4)
		ax2.tick_params('y', colors="slateblue")
		lns = ln1+ln2+ln3+ln4
		labels = [l.get_label() for l in lns]
		ax[2].legend(lns, labels, loc='best', prop={'size': 9})
		ax[2].set_xticks(cruise["Distance_from_station_36"].values[::3])
		ax[2].set_xticklabels(st_names[::3], rotation=45)
		fig.savefig(figpath + "cruise_" + dt.datetime.fromtimestamp(t[d]).strftime('%Y%m%d') + "_temp.png")
	

