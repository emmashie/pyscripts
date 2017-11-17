#### SAME SCRIPT (ish) AS MODEL METRICS, WITHOUT VERTICAL LAYERS, RMS RATIO INSTEAD OF DIFF OF MOD-OBS RMS

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
fname = "adcp_metrics"
 
obs_path = "/opt/data/noaa/ports/"                 
obs = ["SFB1301-2013.nc", "SFB1302-2013.nc", "SFB1304-2013.nc", "SFB1305-2013.nc", 
        "SFB1306-2013.nc", "SFB1307-2013.nc", "SFB1308-2013.nc"]
        
output_path = "/opt/data/delft/sfb_dfm_v2/runs/wy2013a/DFM_OUTPUT_wy2013a/"	
model_files = output_path + "wy2013a_0000_20120801_000000_his.nc"
mdat = nc.MFDataset(model_files)

# pull out utm coordinates of model stations
xcoor = mdat.variables["station_x_coordinate"][:]
ycoor = mdat.variables["station_y_coordinate"][:]

# convert utm coordinates to lat lon 
mll = np.zeros((2, len(xcoor)))
for i in range(len(xcoor)):
	mll[:,i] = utm_to_ll([xcoor[i], ycoor[i]])

# open file to write to 
f = open(path + fname, "w")
# write header line
f.write("\\begin{center}")
f.write("   \\begin{tabular}{| l | l | l | l | l | l | l | l | l |} \n")
f.write("   \\hline \n")
f.write("   Name & Skill [u] & Skill [v] & Bias [u] & Bias [v] & $r^2$ [u] & $r^2$ [v] & RMS [u] & RMS [v] \\\ \\hline \n")
#f.write("filename, skill [u], skill [v], bias [u], bias [v], r2 [u], r2 [v], rms [u], rms [v]\n")
# find model station indicies closest to observation 
llind = np.zeros(len(adcp_files))
for i in range(len(llind)):
	filename = dir + ".txt"
	### create directory for text files ###
	path = output_path + "model_metrics/" 
	if not os.path.exists(path):
		os.makedirs(path)
	dat = nc.Dataset(obs_path + obs[i])
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
		#f.write("%s, avg, %f, %f, %f, %f, %f, %f, %f, %f \n" % (dir, msu, msv, biasu, biasv, r2u, r2v, rmsu, rmsv))
        f.write("   %s & %f & %f & %f & %f & %f & %f & %f & %f \\\ \\hline \n" % (dir, msu, msv, biasu, biasv, r2u, r2v, rmsu, rmsv))
f.write("   \\hline \n")
f.write("   \\end{tabular} \n")
f.write("\\end{center} \n")
f.close()	
      


