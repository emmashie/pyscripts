import numpy as np
import netCDF4 as nc
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib.dates import date2num
from stompy.utils import model_skill
import pyscripts.analysis as an
from stompy.spatial import proj_utils
import os
from stompy.utils import model_skill

ll_to_utm = proj_utils.mapper('WGS84','EPSG:26910')
utm_to_ll = proj_utils.mapper('EPSG:26910','WGS84')

def metric_table(obspath, obs, modpath, mod, metpath, met):
    mdat = nc.MFDataset(modpath+mod)
    mtime = mdat.variables["time"][:]
    mu = mdat.variables["x_velocity"][:]
    mv = mdat.variables["y_velocity"][:]
    mubar = np.mean(mu, axis=-1)
    mvbar = np.mean(mv, axis=-1)
    mdep = mdat.variables["zcoordinate_c"][:]
    # pull out utm coordinates of model stations
    xcoor = mdat.variables["station_x_coordinate"][:]
    ycoor = mdat.variables["station_y_coordinate"][:]
    # convert utm coordinates to lat lon
    mll = np.zeros((2, len(xcoor)))
    for i in range(len(xcoor)):
        mll[:,i] = utm_to_ll([xcoor[i], ycoor[i]])
    # open file to write to 
    f = open(metpath + met, "w")
    # write header lines
    f.write("\\begin{center} \n")
    f.write("\\begin{adjustbox}{width=1\\textwidth} \n")
    f.write("\\begin{tabular}{| l | l | l | l | l | l | l | l | l | l | l |} \n")
    f.write("\\hline \n")
    f.write("Name & Skill (West) & Skill (North) & Bias (West) [m/s] & Bias (North) [m/s] & \(r^2\) (West) & \(r^2\) (North) & RMS (West) [m/s] & RMS (North) [m/s] & Lag (West) [s] & Lag (North) [s]\\\ \\hline \n")
    # find model station indicies closest to observation
    llind = np.zeros(len(obs))
    for i in range(len(llind)):
        dir = str(obs[i][:-3])
        filename = dir + ".txt"
        # create directory for text files 
        if not os.path.exists(metpath):
            os.makedirs(metpath)
        dat = nc.Dataset(obspath + obs[i])
        dist = np.sqrt((mll[0,:]-dat.variables["longitude"][:])**2 + (mll[1,:]-dat.variables["latitude"][:])**2)
        llind[i] = np.where(dist == np.min(dist))[0]
        # define observation variables
        time = dat.variables["time"][:]
        u = dat.variables["u"][:]
        v = dat.variables["v"][:]
        ubar = dat.variables["u_davg"][:]
        vbar = dat.variables["v_davg"][:]
        dep = -dat.variables["depth"][:]
        # convert time to datetime objects and check overlap with observations 
        mdtime = []
        for j in range(len(mtime)):
            mdtime.append(dt.datetime.fromtimestamp(mtime[j] + 15553*86400, tz=dt.timezone.utc))
        # convert datetime objects to datenums to use for interpolation 
        mtimes = date2num(mdtime)
        # take observation time and create datetime objects
        dtime = []
        for j in range(len(time)):
            dtime.append(dt.datetime.fromtimestamp(time[j], tz=dt.timezone.utc))
        # convert datetime objects to datenums to use for interpolation
        dtimes = date2num(dtime)
        # check for overlapping observations and model data
        if dtime[0] < mdtime[0]:
            tmin = mdtime[0]
        else:
            tmin = dtime[0]
        if dtime[-1] < mdtime[-1]:
            tmax = dtime[-1]
        else:
            tmax = mdtime[-1]
        if tmin < tmax:
            mubar_i = np.interp(dtimes, mtimes, mubar[:,llind[i]])
            mvbar_i = np.interp(dtimes, mtimes, mvbar[:,llind[i]])
            msu, biasu, r2u, rmsu, lagu = an.model_metrics(dtimes[(time >= tmin.timestamp()) & (time <= tmax.timestamp())], mubar_i[(time >= tmin.timestamp()) & (time <= tmax.timestamp())], dtimes[(time >= tmin.timestamp()) & (time <= tmax.timestamp())], ubar[(time >= tmin.timestamp()) & (time <= tmax.timestamp())])
            msv, biasv, r2v, rmsv, lagv = an.model_metrics(dtimes[(time >= tmin.timestamp()) & (time <= tmax.timestamp())], mvbar_i[(time >= tmin.timestamp()) & (time <= tmax.timestamp())], dtimes[(time >= tmin.timestamp()) & (time <= tmax.timestamp())], vbar[(time >= tmin.timestamp()) & (time <= tmax.timestamp())])
            f.write("%s & %0.2f & %0.2f & %0.2f & %0.2f & %0.2f & %0.2f & %0.2f & %0.2f & %0.2f & %0.2f \\\ \\hline \n" % (dir[:-5], msu, msv, biasu, biasv, r2u, r2v, rmsu, rmsv, lagu, lagv))
    f.write("\\hline \n")
    f.write("\\end{tabular} \n")
    f.write("\\end{adjustbox} \n")
    f.write("\\end{center} \n")
    f.close()
            
# model output        
modpath = "/opt/data/delft/sfb_dfm_v2/runs/wy2013a/DFM_OUTPUT_wy2013a/"	
mod = "wy2013a_0000_20120801_000000_his.nc"
# metrics output location
metpath = modpath + "model_metrics/"
met = "south_adcp_metrics"
# adcp observations 
obspath = "/opt/data/noaa/ports/"                 
obs = ["SFB1301-2013.nc", "SFB1302-2013.nc", "SFB1304-2013.nc", "SFB1305-2013.nc", 
        "SFB1306-2013.nc", "SFB1307-2013.nc", "SFB1308-2013.nc"]

metric_table(obspath, obs, modpath, mod, metpath, met)

# adcp observations
met = "central_adcp_metrics"
obs = ["SFB1202-2012.nc", "SFB1203-2012.nc", "SFB1204-2012.nc", "SFB1205-2012.nc", 
		"SFB1206-2012.nc", "SFB1207-2012.nc", "SFB1208-2012.nc", "SFB1209-2012.nc",
		"SFB1210-2012.nc", "SFB1211-2012.nc", "SFB1212-2012.nc", "SFB1213-2012.nc",
		"SFB1214-2012.nc", "SFB1215-2012.nc", "SFB1216-2012.nc", "SFB1217-2012.nc",
		"SFB1218-2012.nc", "SFB1219-2012.nc", "SFB1309-2013.nc", "SFB1310-2013.nc",
		"SFB1311-2013.nc", "SFB1312-2013.nc"]

metric_table(obspath, obs, modpath, mod, metpath, met)

# adcp observations
met = "coastal_adcp_metrics"
obs = ["SFB1201-2012.nc", "SFB1220-2012.nc", "SFB1221-2012.nc", "SFB1222-2012.nc",
        "SFB1223-2012.nc"]
        
metric_table(obspath, obs, modpath, mod, metpath, met)

# adcp observations
met = "north_adcp_metrics"
obs = ["SFB1313-2013.nc", "SFB1314-2013.nc", "SFB1315-2013.nc", "SFB1316-2013.nc",
        "SFB1317-2013.nc", "SFB1318-2013.nc", "SFB1319-2013.nc", "SFB1320-2013.nc",
        "SFB1322-2013.nc", "SFB1323-2013.nc", "SFB1324-2013.nc", "SFB1325-2013.nc",
        "SFB1326-2013.nc", "SFB1327-2013.nc", "SFB1328-2013.nc", "SFB1329-2013.nc",
        "SFB1330-2013.nc", "SFB1331-2013.nc", "SFB1332-2013.nc"]
        
metric_table(obspath, obs, modpath, mod, metpath, met)



