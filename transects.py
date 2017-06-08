import xarray as xr
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import cmocean.cm as cmo
import datetime as dt
import pandas as pd

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

### Load in his files, define variables, adjust times to same reference time as USGS data
his = nc.MFDataset("/home/emma/sfb_dfm_setup/r14/DFM_OUTPUT_r14/his_files/r14_0000*.nc")
xcoor = his.variables["station_x_coordinate"][:]
ycoor = his.variables["station_y_coordinate"][:]
dref = 15553*24*60*60
dtz = 7*60*60
mtimes = his.variables["time"][:]+dref+dtz
msalt = his.variables["salinity"][:]
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
mtemp_ = np.zeros((len(tind[:,0]), len(tind[0,:]), len(mdepth[0,0,:])))
mdepth_ = np.zeros((len(tind[:,0]), len(tind[0,:]), len(mdepth[0,0,:])))
for i in range(len(msalt_[:,0,0])):
	for j in range(len(msalt_[0,:,0])):
		msalt_[i,j,:] = msalt[tind[i,j], llind[j], :]
		mtemp_[i,j,:] = mtemp[tind[i,j], llind[j], :]
		mdepth_[i,j,:] = mdepth[tind[i,j], llind[j], :]
		mdates[i,j,:] = mtimes[tind[i,j]]

msalt_ = np.ma.masked_where(msalt_ == -999., msalt_)
mtemp_ = np.ma.masked_where(mtemp_ == -999., mtemp_)
mdepth_ = np.ma.masked_where(mdepth_ == -999., mdepth_)

mdist_ = np.zeros((len(dist), len(mdepth_[0,0,:])))
for i in range(len(mdepth_[0,0,:])):
	mdist_[:,i] = dist[:]

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

for d in range(len(tind[:,0])):
	fig,ax=plt.subplots(nrows=2, figsize=(12,6))
	ax[0].axis('tight')
	cruise=ds.isel(date=d) # choose a single cruise
	field='salinity'
	obs=plot_utils.transect_tricontourf(cruise[field],ax=ax[0],V=np.linspace(0,35,36),
                                     cmap=cmo.matter_r,
                                     xcoord='Distance_from_station_36',
                                     ycoord='depth')
	ax[0].set_ylabel('Depth (m)')
	cbar0 = plt.colorbar(obs,label=field, ax=ax[0])
	ax[0].set_title('cruise# %d' % d)
	ax[1].axis('tight')
	mod=plot_utils.transect_tricontourf(mdat.isel(date=d)[field],ax=ax[1],V=np.linspace(0,35,36),
                                     cmap=cmo.matter_r,
                                     xcoord='Distance_from_station_36',
                                     ycoord='depth')
	ax[1].set_ylabel('Depth (m)')
	cbar1 = plt.colorbar(mod,label=field, ax=ax[1])
	fig.savefig("validation_plots/usgs_transects/" + "cruise_" + str(d) + "_salt.png")
	
	fig,ax=plt.subplots(nrows=2, figsize=(12,6))
	ax[0].axis('tight')
	cruise=ds.isel(date=d) # choose a single cruise
	field='temperature'
	obs=plot_utils.transect_tricontourf(cruise[field],ax=ax[0],V=np.linspace(14,25,36),
                                     cmap='viridis',
                                     xcoord='Distance_from_station_36',
                                     ycoord='depth')
	ax[0].set_ylabel('Depth (m)')
	cbar0 = plt.colorbar(obs,label=field, ax=ax[0])
	ax[0].set_title('cruise# %d' % d)
	ax[1].axis('tight')
	mod=plot_utils.transect_tricontourf(mdat.isel(date=d)[field],ax=ax[1],V=np.linspace(14,25,36),
                                     cmap='viridis',
                                     xcoord='Distance_from_station_36',
                                     ycoord='depth')
	ax[1].set_ylabel('Depth (m)')
	cbar1 = plt.colorbar(mod,label=field, ax=ax[1])
	fig.savefig("validation_plots/usgs_transects/" + "cruise_" + str(d) + "_temp.png")
	

