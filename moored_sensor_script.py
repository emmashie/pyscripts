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
from pyscripts import analysis as an
plt.ioff()

ll_to_utm = proj_utils.mapper('WGS84','EPSG:26910')
utm_to_ll = proj_utils.mapper('EPSG:26910','WGS84')

data_path = os.path.join("/hpcvol1/emma/sfb_dfm/moored_sensor_data")
file = "MooredSensor_L3.nc"
moored_sensor_data_path = os.path.join(data_path,file)
moored_sensor_data = xr.open_dataset(moored_sensor_data_path)

run_name="wy2017-v4"
begindate = "20160801"
path = "/hpcvol1/emma/sfb_dfm/runs/%s/DFM_OUTPUT_%s/"%(run_name,run_name)
hisfile = os.path.join(path, "%s_0000_%s_000000_his.nc"%(run_name,begindate))

start_date = np.datetime64('2016-10-01')
end_date = np.datetime64('2017-10-01')

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
#metric_fn  = os.path.join(metricpath + "moored_sensors.tex")
os.path.exists(savepath) or os.makedirs(savepath)
os.path.exists(metricpath) or os.makedirs(metricpath)

sta = ['ALV', 'DMB', 'POND', 'COY', 'GL', 'MOW', 'NW', 'SM']
sta_names = {'ALV':'Alviso Slough', 'DMB':'Dumbarton Bridge', 
             'POND':'Pond A8', 'COY':'Coyote Creek', 'GL':'Guadalupe Slough', 
             'MOW':'Mowry Slough', 'NW':'Newark Slough', 'SM' : 'San Mateo Bridge'}

fsaltm = open(os.path.join(metricpath,'moored_sensor_salt_metrics.tex'), "w")
# formatting of the table is included but commented out so that minor tweaks
# can be made in the final tex document
fsaltm.write("% \\begin{center} \n")
fsaltm.write("% \\begin{adjustbox}{width=1\\textwidth} \n")
fsaltm.write("% \\begin{tabular}{| l | r | r | r | r | r | r |} \n")
fsaltm.write("% \\hline \n")
fsaltm.write("% Name             & Skill   &  Bias (m) & \(r^2\) & RMSE (m) & Lag (min) & Amp. factor\\\ \\hline \n")

ftempm = open(os.path.join(metricpath,'moored_sensor_temp_metrics.tex'), "w")
# formatting of the table is included but commented out so that minor tweaks
# can be made in the final tex document
ftempm.write("% \\begin{center} \n")
ftempm.write("% \\begin{adjustbox}{width=1\\textwidth} \n")
ftempm.write("% \\begin{tabular}{| l | r | r | r | r | r | r |} \n")
ftempm.write("% \\hline \n")
ftempm.write("% Name             & Skill   &  Bias (m) & \(r^2\) & RMSE (m) & Lag (min) & Amp. factor\\\ \\hline \n")


for s in sta:
    loc = pd.read_csv(os.path.join(data_path, "locations.csv"))
    idx = np.where(loc.Sta.values==s)
    st_ll = [loc.Longitude.values[idx], loc.Latitude.values[idx]]
    dist = (mll[0,:]-st_ll[0])**2 + (mll[1,:]-st_ll[1])**2
    idx = np.argmin(dist) 
    obs = moored_sensor_data[s]   
    mod = mdat.isel(stations=idx)
    #### time and depth
    mtime = mod['salinity'].time.values
    mts = np.asarray([(mtime[i] - np.datetime64('1970-01-01T00:00:00Z'))/np.timedelta64(1,'s') for i in range(len(mtime))])
    mdepth = mod['zcoordinate_c'].values
    otime = obs.time.values
    ots = np.asarray([(otime[i] - np.datetime64('1970-01-01T00:00:00Z'))/np.timedelta64(1,'s') for i in range(len(otime))])
    ots += 8*3600
    odepth = obs.isel(params=9).values
    ## times that are within water year
    oind = np.where((otime>=start_date)&(otime<end_date))[0]
    mind = np.where((mtime>=start_date)&(mtime<end_date))[0]
    ## interpolate obs depth to model times
    fdepth = interpolate.interp1d(ots, odepth,bounds_error=False)
    idepth = fdepth(mts)    
    ### fill nans with average depth to avoid getting bottom depth 
    idepth[np.where(np.isfinite(idepth)==False)[0]] = np.nanmean(idepth) ## possibly fill with harmonic fit? does it matter that much?
    idz = np.array([np.argmin(np.abs(-idepth[i]-mdepth[i,:])) for i in range(len(idepth))])
    #### salinity
    msalt = np.asarray([mod['salinity'].values[i,idz[i]] for i in range(len(idz))]) 
    # interpolate model salt
    fsalt = interpolate.interp1d(mts,msalt,bounds_error=False)
    isalt = fsalt(ots)   
    osalt = obs.isel(params=11).values
    valid = np.where(np.isfinite(isalt))[0]
    salt_ms, salt_bias, salt_r2, salt_rms, salt_lag, salt_amp = an.model_metrics(ots[valid], isalt[valid],
                                                                                 ots[valid], osalt[valid])

    fsaltm.write("%-16s & %7.3f & %9s & %7.3f &  %7.3f &    %6.1f &   %6.2f   \\\ \\hline \n" % (sta_names[s],
                                                                                            salt_ms, salt_bias, salt_r2, 
                                                                                            salt_rms, salt_lag, salt_amp))

    fig, ax = plt.subplots(figsize=(10,4.75))
    ax.plot(otime[oind], osalt[oind], color='cornflowerblue', label='Observations')
    ax.plot(mtime[mind], msalt[mind], color='k', alpha=0.7, label='Model')
    ax.legend(loc='best')
    ax.set_title('%s: Salinity' % (sta_names[s]))
    fig.savefig('%s%s_salt.png' % (savepath, sta_names[s]))
    #### temperature 
    mtemp = np.asarray([mod['temperature'].values[i,idz[i]] for i in range(len(idz))]) 
    # interpolate model salt
    ftemp = interpolate.interp1d(mts,mtemp,bounds_error=False)
    itemp = ftemp(ots)   
    otemp = obs.isel(params=0).values
    valid = np.where(np.isfinite(itemp))[0]
    temp_ms, temp_bias, temp_r2, temp_rms, temp_lag, temp_amp = an.model_metrics(ots[valid], itemp[valid],
                                                                                 ots[valid], otemp[valid])

    ftempm.write("%-16s & %7.3f & %9s & %7.3f &  %7.3f &    %6.1f &   %6.2f   \\\ \\hline \n" % (sta_names[s],
                                                                                            temp_ms, temp_bias, temp_r2, 
                                                                                            temp_rms, temp_lag, temp_amp))
    fig, ax = plt.subplots(figsize=(10,4.75))
    ax.plot(otime[oind], otemp[oind], color='cornflowerblue', label='Observations')
    ax.plot(mtime[mind], mtemp[mind], color='k', alpha=0.7, label='Model')
    ax.legend(loc='best')
    ax.set_title('%s: Temperature' % (sta_names[s]))
    fig.savefig('%s%s_temp.png' % (savepath, sta_names[s]))
    plt.close('all')

fsaltm.close()
ftempm.close()