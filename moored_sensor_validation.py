import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import num2date,date2num
from datetime import datetime, timedelta
from matplotlib.dates import DateFormatter
from scipy.stats.stats import pearsonr
from scipy import stats
import os 
from stompy.spatial import proj_utils
import xarray as xr 
from scipy import interpolate

ll_to_utm = proj_utils.mapper('WGS84','EPSG:26910')
utm_to_ll = proj_utils.mapper('EPSG:26910','WGS84')

data_path = os.path.join("/hpcvol1/emma/sfb_dfm/moored_sensor_data")

run_name="wy2017-v4"
begindate = "20160801"
path = "/hpcvol1/emma/sfb_dfm/runs/%s/DFM_OUTPUT_%s/"%(run_name,run_name)
hisfile = os.path.join(path, "%s_0000_%s_000000_his.nc"%(run_name,begindate))

mdat = xr.open_dataset(hisfile)
# pull out coordinates of model stations
xcoor = mdat["station_x_coordinate"].values
ycoor = mdat["station_y_coordinate"].values
# convert utm coordinates to lat lon
xcoor=xcoor[0,:]
ycoor=ycoor[0,:]
mll = utm_to_ll(np.c_[xcoor,ycoor]).T

savepath = os.path.join(path + "validation_plots/moored_sensors/")
metricpath = os.path.join(path + "validation_metrics/")
metric_fn  = os.path.join(metricpath + "moored_sensors.tex")
os.path.exists(savepath) or os.makedirs(savepath)
os.path.exists(metricpath) or os.makedirs(metricpath)


sta = ['ALV', 'DMB', 'POND', 'COY', 'GL', 'MOW', 'NW', 'SM']
sta_cols = {'ALV':'goldenrod', 'DMB':'orchid', 'POND':'brown', 
            'COY':'olive', 'GL':'tomato', 'MOW':'DarkSeaGreen', 
            'NW':'steelblue', 'SM' : "#3F5D7D"}
sta_names = {'ALV':'Alviso Slough', 'DMB':'Dumbarton Bridge', 
             'POND':'Pond A8', 'COY':'Coyote Creek', 'GL':'Guadalupe Slough', 
             'MOW':'Mowry Slough', 'NW':'Newark Slough', 'SM' : 'San Mateo Bridge'}
var = ['T_degC', 'S_PSU', 'Depth_m']
var_names = {'T_degC' : 'Temp (C)', 'S_PSU' : 'Salinity (PSU)', 'Depth_m' : 'Depth (m)'}

start = 1470009600.0; end = 1506812400.0
dstart = datetime(2016,8,1,0,0,0); dend = datetime(2017,9,30,23,0,0)  

for s in sta:
    df = pd.read_csv(os.path.join(data_path, "%s" % (s) + "_all_data_L3.csv"))
    df['dt'] = pd.to_datetime(df['dt']) 
    dt = [df['dt'].values[i] + np.timedelta64(8,'h') for i in range(len(df['dt'].values))]
    ts_obs = np.array([(dt[i] - np.datetime64('1970-01-01T00:00:00Z'))/np.timedelta64(1,'s') for i in range(len(dt))])
    mask = (df['dt'] >= dstart) & (df['dt'] <= dend)  
    df = df[mask] 
    mask = (ts_obs >= start) & (ts_obs <= end)  
    ts_obs = ts_obs[mask]
    loc = pd.read_csv(os.path.join(data_path, "locations.csv"))
    idx = np.where(loc.Sta.values==s)
    st_ll = [loc.Longitude.values[idx], loc.Latitude.values[idx]]
    temp_obs = df[var[0]].values[:len(ts_obs)]
    salt_obs = df[var[1]].values[:len(ts_obs)] 
    depth_obs = df[var[2]].values[:len(ts_obs)]
    dist = (mll[0,:]-st_ll[0])**2 + (mll[1,:]-st_ll[1])**2
    idx = np.argmin(dist)
    temp = mdat.temperature[:,idx,:].values
    salt = mdat.salinity[:,idx,:].values
    depth = mdat.zcoordinate_c[:,idx,:].values
    ts = [(mdat.time.values[i] - np.datetime64('1970-01-01T00:00:00Z'))/np.timedelta64(1,'s') for i in range(len(mdat.time.values))]
    itemp = np.zeros((len(ts_obs),10))
    isalt = np.zeros((len(ts_obs),10))
    idepth = np.zeros((len(ts_obs),10))
    for i in range(len(depth[0,:])):
        ftemp = interpolate.interp1d(ts, temp[:,i])
        itemp[:,i] = ftemp(ts_obs)
        fsalt = interpolate.interp1d(ts, salt[:,i])
        isalt[:,i] = fsalt(ts_obs)
        fdepth = interpolate.interp1d(ts, depth[:,i])
        idepth[:,i] = fdepth(ts_obs)
    #for i in range(len(idepth[:,0])):
    #    idepth[i,:] = idepth[i,:] - idepth[i,0]
    zind = np.array([np.argmin(np.abs(idepth[i,:]-depth_obs[i])) for i in range(len(idepth[:,0]))])
    temp = np.array([itemp[i,zind[i]] for i in range(len(zind))])
    salt = np.array([isalt[i,zind[i]] for i in range(len(zind))])
    fig, ax = plt.subplots(nrows=2, figsize=(10,8), sharex=True)
    ax[0].plot(df['dt'][:len(ts_obs)], temp_obs, label="Obs")
    ax[0].plot(df['dt'][:len(ts_obs)], temp, color='black', label="Model")
    ax[0].set_ylabel("%s" % (var_names[var[0]]))
    ax[0].legend(loc='best')
    ax[1].plot(df['dt'][:len(ts_obs)], salt_obs, label="Obs")
    ax[1].plot(df['dt'][:len(ts_obs)], salt, color='black', label="Model")
    ax[1].set_ylabel("%s" % (var_names[var[1]]))    
    ax[1].legend(loc='best')
    fig.autofmt_xdate()
    ax[0].set_title("%s" % (sta_names[s]))
    #fig.savefig("%s%s_noaltz.png" % (savepath, sta_names[s]))
    #plt.close('all')











