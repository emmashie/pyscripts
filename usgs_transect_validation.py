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
import stompy.plot.cmap as scmap
from stompy.io.local import usgs_sfbay
from stompy.spatial import proj_utils
import stompy.model.delft.io as dio

ll_to_utm = proj_utils.mapper('WGS84','EPSG:26910')
utm_to_ll = proj_utils.mapper('EPSG:26910','WGS84')


##

# Set the model location
run_name="wy2013c"
path = "/opt/data/delft/sfb_dfm_v2/runs/%s/DFM_OUTPUT_%s/"%(run_name,run_name)
hisfile = "%s_0000_20120801_000000_his.nc"%run_name
mdu=dio.MDUFile(os.path.join(path,'../%s.mdu'%run_name))
cache_dir=os.path.join(path,"validation_metrics/cache")
os.path.exists(cache_dir) or os.makedirs(cache_dir)

### Load in USGS cruise data, define variables, and convert datetime64 times to timestamps

usgs_cache_fn=os.path.join(cache_dir,'usgs_cruises.nc')
if not os.path.exists(usgs_cache_fn):
    ds=usgs_sfbay.cruise_dataset(np.datetime64('2012-08-01'), np.datetime64('2013-10-01'))
    ds.to_netcdf(usgs_cache_fn)
    ds.close()
# clean read:
ds=xr.open_dataset(usgs_cache_fn)

## 

lat = np.asarray(ds["latitude"])
lon = np.asarray(ds["longitude"])
salt = np.asarray(ds["salinity"])
temp = np.asarray(ds["temperature"])
depth = np.asarray(ds["depth"])
dist = np.asarray(ds["Distance_from_station_36"])
time = np.asarray(ds["time"])
# These times already come in as UTC. the date math below does everything in
# PST, so take the 8 hours back in 
times = -8*3600 + (time - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 's')

### fix nans in time record and fill in with valid times
tvalid = utils.fill_invalid(times, axis=-1)
tvalid = utils.fill_invalid(tvalid, axis=1)
tvalid = utils.fill_invalid(tvalid, axis=0)
times = tvalid

### Load in his files, define variables, adjust times to same reference time as USGS data

temp = False

his = nc.MFDataset(path + hisfile)
xcoor = his.variables["station_x_coordinate"][:]
ycoor = his.variables["station_y_coordinate"][:]

# get reference time from the mdu:
t_ref,t_start,t_stop=mdu.time_range()
dref=utils.to_unix(t_ref) # dt.datetime.strptime(emdu['time','refdate']

dtz = -8*60*60 # adjust UTC to PST
mtimes = his.variables["time"][:]+dref+dtz
msalt = his.variables["salinity"][:]
if temp == True:
    mtemp = his.variables["temperature"][:]
mdepth = his.variables["zcoordinate_c"][:]

if xcoor.ndim==2:
    # some DFM runs add time to coordinates
    xcoor=xcoor[0,:]
    ycoor=ycoor[0,:]
    
mll = utm_to_ll(np.c_[xcoor, ycoor]).T

llind = np.zeros(len(ds["latitude"]),np.int32)
for i in range(len(llind)):
    dist_ = np.sqrt((mll[0,:]-lon[i])**2 +  (mll[1,:]-lat[i])**2)
    llind[i] = np.argmin(dist_) # np.where(dist_ == np.min(dist_))[0]

tind = np.zeros(times[:,:,0].shape,np.int32)
for i in range(tind.shape[0]): # loop over cruises
    for j in range(tind.shape[1]): # loop over stations within cruise
        ind=utils.nearest(mtimes, times[i,j,0]) # match on time at top of cast
        # if the match is more than a day off, assume the model
        # doesn't cover this cruise.
        if np.abs(mtimes[ind]-times[i,j,0])>86400:
            ind=-1
        tind[i,j]=ind

mdates = np.zeros((len(tind[:,0]), len(tind[0,:]), len(mdepth[0,0,:])))
msalt_ = np.zeros((len(tind[:,0]), len(tind[0,:]), len(mdepth[0,0,:])))
if temp == True:
    mtemp_ = np.zeros((len(tind[:,0]), len(tind[0,:]), len(mdepth[0,0,:])))
mdepth_ = np.zeros((len(tind[:,0]), len(tind[0,:]), len(mdepth[0,0,:])))

for i in range(len(msalt_[:,0,0])):
    for j in range(len(msalt_[0,:,0])):
        if tind[i,j]>=0:
            msalt_[i,j,:] = msalt[tind[i,j], llind[j], :]
            mdepth_[i,j,:] = mdepth[tind[i,j], llind[j], :]
            mdates[i,j,:] = mtimes[tind[i,j]]
            if temp == True:
                mtemp_[i,j,:] = mtemp[tind[i,j], llind[j], :]
        else:
            msalt_[i,j,:] = np.nan
            mdepth_[i,j,:] = np.nan
            mdates[i,j,:] = np.nan
            if temp == True:
                mtemp_[i,j,:] = np.nan

msalt_ = np.ma.masked_where(msalt_ == -999., msalt_)
if temp == True:
    mtemp_ = np.ma.masked_where(mtemp_ == -999., mtemp_)
mdepth_ = np.ma.masked_where(mdepth_ == -999., mdepth_)

mdist_ = np.zeros((len(dist), len(mdepth_[0,0,:])))
for i in range(len(mdepth_[0,0,:])):
    mdist_[:,i] = dist[:]


coords={'date': (['date'], np.arange(len(tind))),
        'Distance_from_station_36': (['Distance_from_station_36'], dist), 
        'prof_sample': (['prof_sample'], np.arange(10)),
        'depth': (['date', 'Distance_from_station_36', 'prof_sample'], mdepth_),
        'times': (['date', 'Distance_from_station_36', 'prof_sample'], mdates)}
mdat = xr.Dataset({'salinity': (['date', 'Distance_from_station_36', 'prof_sample'], msalt_)},
                      coords=coords)
if temp:
    mdat['temperature'] = ('date', 'Distance_from_station_36', 'prof_sample'), mtemp_
    
if 1: # Take out the freesurface droop, which is a kind of distracting
    # Freesurface droop is at least from the tidal variations in freesurface
    # over the cruise, and probably also includes some sigma-coordinate confusion.
    # It's slightly wrong to zero this out without worrying about the sigma-coordinate
    # stuff, but overall it's a fairly qualitative figure anyway
    mdat.depth.values[:] = mdat.depth - mdat.depth.max(dim='prof_sample')
    
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

##

# occasionally there is a just a single profile sample in a cast
# this disrupts the contour plots.  Set this to 2 or more to
# drop those profiles
min_samples_per_profile=2

salt_cmap=cmo.matter_r

salt_contours=np.arange(5,35,5)

# avg times for plotting and figure names
t = np.nanmean(np.nanmean(times,axis=-1),axis=-1)
for d in range(len(tind[:,0])):
    # counter-intuitive - but because above we already took 8 hours off
    # to put t[:] into PST, here we tell python it's UTC, so that python
    # doesn't try to further correct for timezones.  utcfromtimestamp doesn't
    # set a time zone on the return value, and doesn't try to make any
    # time zone adjustments
    cruise_start=dt.datetime.utcfromtimestamp(t[d])
    cruise_label=cruise_start.strftime('%Y-%m-%d')    

    if np.all(tind[d,:]<0):
        print("Cruise %s is not covered by model run"%cruise_label)
        continue
    
    fig,ax=plt.subplots(nrows=3, figsize=(8.5,9.5), sharex=True)
    fig.subplots_adjust(bottom=0.14)
    
    ax[0].axis('tight')
    cruise=ds.isel(date=d) # choose a single cruise
    field='salinity'
    field_label='Salinity [ppt]'

    bad_profiles = np.isfinite(cruise.salinity.values).sum(axis=1) < min_samples_per_profile
    cleaned=cruise[field].copy()
    # Just set the whole column to nan, and it will get dropped from the contour plots.
    cleaned.loc[dict(Distance_from_station_36=bad_profiles)] = np.nan
    
    obs=plot_utils.transect_tricontourf(cleaned,ax=ax[0],V=np.linspace(0,35,36),
                                        cmap=salt_cmap,
                                        xcoord='Distance_from_station_36',
                                        ycoord='depth',
                                        extend='max')
    obsl=plot_utils.transect_tricontour(cleaned,ax=ax[0],V=salt_contours,
                                        colors='k',linewidths=0.8,alpha=0.5,
                                        xcoord='Distance_from_station_36',
                                        ycoord='depth')
    
    ax[0].set_ylabel('Depth (m)')

    ax0_pos=ax[0].get_position()
    cbar0 = plt.colorbar(obs,label=field_label,
                         cax=plt.gcf().add_axes((0.93,ax0_pos.ymin,0.01,ax0_pos.height)))
    ax[0].set_title('Cruise %s' % cruise_label)
    ax[1].axis('tight')
    mod=plot_utils.transect_tricontourf(mdat.isel(date=d)[field],ax=ax[1],V=np.linspace(0,35,36),
                                        cmap=salt_cmap,
                                        xcoord='Distance_from_station_36',
                                        ycoord='depth',
                                        extend='max')
    modl=plot_utils.transect_tricontour(mdat.isel(date=d)[field],ax=ax[1],V=salt_contours,
                                        colors='k',linewidths=0.8,alpha=0.5,
                                        xcoord='Distance_from_station_36',
                                        ycoord='depth')
    
    ax[1].set_ylabel('Depth (m)')
    ax1_pos=ax[1].get_position()
    cbar1 = plt.colorbar(mod,label=field_label, cax=plt.gcf().add_axes((0.93,ax1_pos.ymin,0.01,ax1_pos.height)))
    ln1 = ax[2].plot(mdat.isel(date=d)["Distance_from_station_36"].values, np.nanmean(mdat.isel(date=d)[field].values, axis=-1), 'o-', label="Model - Depth Avg", color="lightseagreen", markersize=4)
    ln2 = ax[2].plot(cruise["Distance_from_station_36"].values, np.nanmean(cruise[field].values, axis=-1), '^--', label="Obs - Depth Avg", color="lightseagreen", markersize=4)
    ax[2].tick_params('y', colors="lightseagreen")
    ax[2].set_ylabel("Salinity [ppt]", color="lightseagreen")
    ax2 = ax[2].twinx()
    mdat_cruise=mdat.isel(date=d)
    mstrat=np.nanmax(mdat_cruise[field],axis=-1) - np.nanmin(mdat_cruise[field],axis=-1)
    ostrat=np.nanmax(cruise[field].values,axis=-1) - np.nanmin(cruise[field].values,axis=-1)

    if 1: # get a "real" ds/dz
        mdz=np.nanmax(mdat_cruise['depth'],axis=-1) - np.nanmin(mdat_cruise['depth'],axis=-1)
        mstrat=mstrat/mdz
        
        odz=np.nanmax(cruise['depth'].values,axis=-1) - np.nanmin(cruise['depth'].values,axis=-1)
        ostrat=ostrat/odz
    
    ln3 = ax2.plot(mdat.isel(date=d)["Distance_from_station_36"].values,
                   mstrat,
                   'o-', label="Model - Stratification", color="slateblue", alpha=1., markersize=4)
    
    ln4 = ax2.plot(cruise["Distance_from_station_36"].values,
                   ostrat,
                   '^--', label="Obs - Stratification", color="slateblue", alpha=1., markersize=4)
    ax2.tick_params('y', colors="slateblue")
    ax2.set_ylabel("Stratification (ds/dz) [ppt/m]", color="slateblue")
    lns = ln1+ln2+ln3+ln4
    labels = [l.get_label() for l in lns]
    ax[2].legend(lns, labels, loc='best', prop={'size': 9})
    ax[2].set_xticks(cruise["Distance_from_station_36"].values[::3])

    ax[0].axis(xmin=-2,xmax=147)

    xxyy0=ax[0].axis()
    xxyy1=ax[1].axis()
    ax[0].axis(ymin=min(xxyy0[2],xxyy1[2]),
               ymax=max(xxyy0[3],xxyy1[3]))
    ax[2].set_xticklabels(st_names[::3], rotation=45,ha='right')

    ax[0].text(0.02,0.05,"Observed",transform=ax[0].transAxes,fontsize=12)
    ax[1].text(0.02,0.05,"Model",transform=ax[1].transAxes,fontsize=12)

    fig.savefig(figpath + "cruise_" + cruise_start.strftime('%Y%m%d') + "_salt.png")
    
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
        ax[0].set_title('cruise# %s' % cruise_start.strftime('%Y-%m-%d'))
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
        fig.savefig(figpath + "cruise_" + cruise_start.strftime('%Y%m%d') + "_temp.png")
    plt.close('all')


##

