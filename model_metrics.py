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

# make list of file name , perhaps make more automated ?
adcp_files = ["SFB1201-2012.nc", "SFB1202-2012.nc", "SFB1203-2012.nc", "SFB1204-2012.nc",
	      "SFB1205-2012.nc", "SFB1206-2012.nc", "SFB1207-2012.nc", "SFB1208-2012.nc",
	      "SFB1209-2012.nc", "SFB1210-2012.nc", "SFB1211-2012.nc", "SFB1212-2012.nc",
	      "SFB1213-2012.nc", "SFB1214-2012.nc", "SFB1215-2012.nc", "SFB1216-2012.nc",
	      "SFB1217-2012.nc", "SFB1218-2012.nc", "SFB1219-2012.nc", "SFB1220-2012.nc",
	      "SFB1221-2012.nc", "SFB1222-2012.nc", "SFB1223-2012.nc", "SFB1301-2013.nc",
	      "SFB1302-2013.nc", "SFB1304-2013.nc", "SFB1305-2013.nc", "SFB1306-2013.nc",
	      "SFB1307-2013.nc", "SFB1308-2013.nc", "SFB1309-2013.nc", "SFB1310-2013.nc",
	      "SFB1311-2013.nc", "SFB1312-2013.nc", "SFB1313-2013.nc", "SFB1314-2013.nc",
	      "SFB1315-2013.nc", "SFB1316-2013.nc", "SFB1317-2013.nc", "SFB1318-2013.nc",
	      "SFB1319-2013.nc", "SFB1320-2013.nc", "SFB1322-2013.nc", "SFB1323-2013.nc",
	      "SFB1324-2013.nc", "SFB1325-2013.nc", "SFB1326-2013.nc", "SFB1327-2013.nc",
	      "SFB1328-2013.nc", "SFB1329-2013.nc", "SFB1330-2013.nc", "SFB1331-2013.nc",
	      "SFB1332-2013.nc"]

# define model files and load netcdf files
#output_path = "/home/emma/sfb_dfm_setup/r14/DFM_OUTPUT_r14/his_files/"			
#model_files = output_path + "r14_0000_201*.nc"
output_path = "/opt/data/delft/sfb_dfm_v2/runs/wy2013/DFM_OUTPUT_wy2013/"	
model_files = output_path + "wy2013_0000_20120801_000000_his.nc"
mdat = nc.MFDataset(model_files)

# pull out utm coordinates of model stations
xcoor = mdat.variables["station_x_coordinate"][:]
ycoor = mdat.variables["station_y_coordinate"][:]

# convert utm coordinates to lat lon 
mll = np.zeros((2, len(xcoor)))
for i in range(len(xcoor)):
	mll[:,i] = utm_to_ll([xcoor[i], ycoor[i]])
	
# find model station indicies closest to observation 
llind = np.zeros(len(adcp_files))
for i in range(len(llind)):
	dir = str(adcp_files[i][:-3])
	filename = dir + ".txt"
	### create directory for text files ###
	path = output_path + "model_metrics/" 
	if not os.path.exists(path):
		os.makedirs(path)
	# open file to write to 
	f = open(path + dir, "w")
	# write header line
	f.write("filename, depth, skill [u], skill [v], bias [u], bias [v], r2 [u], r2 [v], rms [u], rms [v]\n")
	dat = nc.Dataset("/opt/data/noaa/ports/" + adcp_files[i])
	dist = np.sqrt((mll[0,:]-dat.variables["longitude"][:])**2 +  (mll[1,:]-dat.variables["latitude"][:])**2)
	llind[i] = np.where(dist == np.min(dist))[0]
	# define observation variables
	time = dat.variables["time"][:]
	u = dat.variables["u"][:]
	v = dat.variables["v"][:]
	ubar = dat.variables["u_davg"][:]
	vbar = dat.variables["v_davg"][:]
	dep = -dat.variables["depth"][:]
	# define model variables
	mtime = mdat.variables["time"][:]
	mu = mdat.variables["x_velocity"][:]
	mv = mdat.variables["y_velocity"][:]
	mubar = np.mean(mu, axis=-1)
	mvbar = np.mean(mv, axis=-1)
	mdep = mdat.variables["zcoordinate_c"][:]
	# take model time (referenced to 2012-08-01) and create datetime objects to use for plotting
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
	if  tmin < tmax:
		####### DEPTH AVERAGED #######
		# interpolate model to observation times
		mubar_i = np.interp(dtimes, mtimes, mubar[:,llind[i]])
		mvbar_i = np.interp(dtimes, mtimes, mvbar[:,llind[i]])	
		msu, biasu, r2u, rmsu = an.model_metrics(mubar_i[(time >= tmin.timestamp()) & (time <= tmax.timestamp())], ubar[(time >= tmin.timestamp()) & (time <= tmax.timestamp())])
		msv, biasv, r2v, rmsv = an.model_metrics(mvbar_i[(time >= tmin.timestamp()) & (time <= tmax.timestamp())], vbar[(time >= tmin.timestamp()) & (time <= tmax.timestamp())])
		f.write("%s, avg, %f, %f, %f, %f, %f, %f, %f, %f \n" % (dir, msu, msv, biasu, biasv, r2u, r2v, rmsu, rmsv))
		####### VERTICAL LEVELS #######
		# interpolate model to observation depths
		mu_i = np.zeros((len(dep), len(mdtime)))
		mv_i = np.zeros((len(dep), len(mdtime)))
		for j in range(len(mdtime)):
			mu_i[:,j] = np.interp(dep, mdep[j,llind[i],:], mu[j,llind[i],:])
			mv_i[:,j] = np.interp(dep, mdep[j,llind[i],:], mv[j,llind[i],:])
		# interpolate model to observation times
		mu_ii = np.zeros((len(dep), len(dtime)))
		mv_ii = np.zeros((len(dep), len(dtime)))
		for k in range(len(dep)):
			mu_ii[k,:] = np.interp(dtimes, mtimes, mu_i[k,:])
			mv_ii[k,:] = np.interp(dtimes, mtimes, mv_i[k,:])
			# check for nans in observations
			dumu = u[k,:]
			dumum = mu_ii[k,:]
			unan = dumu[~np.isnan(dumu)]
			munan = dumum[~np.isnan(dumu)]
			utime = time[~np.isnan(dumu)]
			dumv = v[k,:]
			dumvm = mv_ii[k,:]
			vnan = dumv[~np.isnan(dumv)]
			mvnan = dumvm[~np.isnan(dumv)]
			vtime = time[~np.isnan(dumv)]
			msu, biasu, r2u, rmsu = an.model_metrics(munan[(utime >= tmin.timestamp()) & (utime <= tmax.timestamp())], unan[(utime >= tmin.timestamp()) & (utime <= tmax.timestamp())])
			msv, biasv, r2v, rmsv = an.model_metrics(mvnan[(vtime >= tmin.timestamp()) & (vtime <= tmax.timestamp())], vnan[(vtime >= tmin.timestamp()) & (vtime <= tmax.timestamp())])
			f.write("%s, %f, %f, %f, %f, %f, %f, %f, %f, %f \n" % (dir, dep[k], msu, msv, biasu, biasv, r2u, r2v, rmsu, rmsv))
	f.close()	
