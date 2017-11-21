import numpy as np
import netCDF4 as nc
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib.dates import date2num
from stompy.utils import model_skill
from stompy.utils import rotate_to_principal
import pyscripts.analysis as an
from stompy.spatial import proj_utils
import os

ll_to_utm = proj_utils.mapper('WGS84','EPSG:26910')
utm_to_ll = proj_utils.mapper('EPSG:26910','WGS84')


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


#output_path = "/home/emma/sfb_dfm_setup/r14/DFM_OUTPUT_r14/his_files/"			
#model_files = output_path + "r14_0000_201*.nc"
output_path = "/opt/data/delft/sfb_dfm_v2/runs/wy2013a/DFM_OUTPUT_wy2013a/"	
model_files = output_path + "wy2013a_0000_20120801_000000_his.nc"
mdat = nc.MFDataset(model_files)

# pull out coordinates of model stations
xcoor = mdat.variables["station_x_coordinate"][:]
ycoor = mdat.variables["station_y_coordinate"][:]
# convert utm coordinates to lat lon 
mll = np.zeros((2, len(xcoor)))
for i in range(len(xcoor)):
	mll[:,i] = utm_to_ll([xcoor[i], ycoor[i]])
	
# get model station indicies for closest station to observation 
llind = np.zeros(len(adcp_files))
for i in range(len(llind)):
	dir = str(adcp_files[i][:-3])
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
		### create directory for figures ###
		path = output_path + "validation_plots/adcp_validation_plots/" + dir
		if not os.path.exists(path):
			os.makedirs(path)
		ts = path + "/time_series"
		if not os.path.exists(ts):
			os.makedirs(ts)
		spec = path + "/spectra"
		if not os.path.exists(spec):
			os.makedirs(spec)
		scat = path + "/scatter"
		if not os.path.exists(scat):
			os.makedirs(scat)		
		####### DEPTH AVERAGED #######
		# interpolate model to observation times
		mubar_i = np.interp(dtimes, mtimes, mubar[:,llind[i]])
		mvbar_i = np.interp(dtimes, mtimes, mvbar[:,llind[i]])
		# rotating model and obs to principle axis
		muvbar = rotate_to_principal(np.asarray([mubar_i, mvbar_i]))
		uvbar = rotate_to_principal(np.asarray([ubar, vbar]))
		# computing observation spectra
		oufreq, ouspec = an.band_avg(time, ubar, dt=(time[1]-time[0])/3600)
		ovfreq, ovspec = an.band_avg(time, vbar, dt=(time[1]-time[0])/3600) 
		#oufreq, ouspec = an.band_avg(time, uvbar[0,:], dt=(time[1]-time[0])/3600)
		#ovfreq, ovspec = an.band_avg(time, uvbar[1,:], dt=(time[1]-time[0])/3600) 
		# computing model spectra
		mufreq, muspec = an.band_avg(mtimes, mubar[:,llind[i]])
		mvfreq, mvspec = an.band_avg(mtimes, mvbar[:,llind[i]])
		#mufreq, muspec = an.band_avg(time, muvbar[0,:], dt=(time[1]-time[0])/3600)
		#mvfreq, mvspec = an.band_avg(time, muvbar[1,:], dt=(time[1]-time[0])/3600)
		# plotting up model & observation time series	
		fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(8,7))
		ax[0].plot(mdtime[:], mubar[:,llind[i]], color='lightslategray', label='Model')
		ax[0].plot(dtime[:], ubar[:], '--', color='cadetblue', label='ADCP')
		#ax[0].plot(dtime[:], muvbar[0,:], color='lightcoral', label='Model')
		#ax[0].plot(dtime[:], uvbar[0,:], color='turquoise', label='ADCP')
		ax[0].legend()
		ax[0].set_title(dir[:-5])
		ax[0].set_ylabel("ubar [m/s]")
		ax[1].plot(mdtime[:], mvbar[:,llind[i]], color='lightslategray')
		ax[1].plot(dtime[:], vbar[:], '--', color='cadetblue')
		#ax[1].plot(dtime[:], muvbar[1,:], color='lightcoral', label='Model')
		#ax[1].plot(dtime[:], uvbar[1,:], color='turquoise', label='ADCP')
		#ax[1].set_xlim((tmin, tmax))
		ax[1].set_xlim((tmin, tmin+dt.timedelta(days=7)))
		ax[1].set_ylabel("vbar [m/s]")
		fig.autofmt_xdate(rotation=45)
		fig.savefig(ts + "/" + adcp_files[i][:-3] + "_time_series.png")
		# plot model & observation spectra
		fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(5,8))
		ax[0].loglog(mufreq, muspec, color='lightcoral', label='Model')
		ax[0].loglog(oufreq, ouspec, 'turquoise', label='ADCP')
		ax[0].legend(loc='best')
		ax[0].set_ylabel("ubar spectral energy [$(m/s)^2/cph$]")
		ax[0].set_title(dir[:-5])
		ax[1].loglog(mvfreq, mvspec, color='lightcoral')
		ax[1].loglog(ovfreq, ovspec, 'turquoise')
		ax[1].set_xlabel("frequency [$cph$]")	
		ax[1].set_xlim((10**-4, 0.5))
		ax[1].set_ylabel("vbar spectral energy [$(m/s)^2/cph$]")
		fig.savefig(spec + "/" + adcp_files[i][:-3] + "_spectra.png")
		# plot scatter of model & observations
		fig, ax = plt.subplots(figsize=(5,3.9))
		#ax.scatter(mubar[:,llind[i]], mvbar[:,llind[i]], s=0.5, alpha=0.5, color='lightcoral', label='Model')
		#ax.scatter(ubar, vbar, s=0.5, alpha=0.5, color='turquoise', label='ADCP')
		mn = np.min([np.nanmin(muvbar[0,:]), np.nanmin(uvbar[0,:])])
		mx = np.max([np.nanmax(muvbar[0,:]), np.nanmax(uvbar[0,:])])
		lin = np.linspace(mn,mx)
		ax.plot(lin,lin, color="lightskyblue", alpha=0.6)
		ind = np.where(muvbar[0,:]!=0)
		ax.scatter(muvbar[0,(muvbar[0,:]!=0) & (uvbar[0,:]!=0)], uvbar[0,(muvbar[0,:]!=0) & (uvbar[0,:]!=0)], s=2, color='lightslategray')
		ax.set_xlim((mn,mx))
		ax.set_ylim((mn,mx))
		#lgnd = ax.legend()
		#lgnd.legendHandles[0]._sizes = [5]
		#lgnd.legendHandles[1]._sizes = [5]
		#ax.set_xlabel("u")
		#ax.set_ylabel("v")
		ax.set_xlabel("model")
		ax.set_ylabel("obs")
		ax.set_title(dir[:-5])
		fig.savefig(scat + "/" + adcp_files[i][:-3] + "_scatter.png")
		####### VERTICAL LEVELS #######
		# interpolate model to observation depths
#		mu_i = np.zeros((len(dep), len(mdtime)))
#		mv_i = np.zeros((len(dep), len(mdtime)))
#		for j in range(len(mdtime)):
#			mu_i[:,j] = np.interp(dep, mdep[j,llind[i],:], mu[j,llind[i],:])
#			mv_i[:,j] = np.interp(dep, mdep[j,llind[i],:], mv[j,llind[i],:])
#		# interpolate model to observation times
#		mu_ii = np.zeros((len(dep), len(dtime)))
#		mv_ii = np.zeros((len(dep), len(dtime)))
#		for k in range(len(dep)):
#			mu_ii[k,:] = np.interp(dtimes, mtimes, mu_i[k,:])
#			mv_ii[k,:] = np.interp(dtimes, mtimes, mv_i[k,:])
#			# check for nans in observations
#			dumu = u[k,:]
#			unan = dumu[~np.isnan(dumu)]
#			utime = time[~np.isnan(dumu)]
#			dumv = v[k,:]
#			vnan = dumv[~np.isnan(dumv)]
#			vtime = time[~np.isnan(dumv)]
#			# computing observation spectra
#			oufreq, ouspec = an.band_avg(utime, unan, dt=(time[1]-time[0])/3600)
#			ovfreq, ovspec = an.band_avg(vtime, vnan, dt=(time[1]-time[0])/3600) 
#			# computing model spectra
#			mufreq, muspec = an.band_avg(mtimes, mu_i[k,:])
#			mvfreq, mvspec = an.band_avg(mtimes, mv_i[k,:])
#			# plotting up model & observation time series	
#			fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(12,6))
#			ax[0].plot(mdtime[:], mu_i[k,:], color='lightcoral', label='Model')
#			ax[0].plot(dtime[:], u[k,:], color='turquoise', label='ADCP')
#			ax[0].legend()
#			ax[0].set_ylabel("u [m/s]")
#			ax[1].plot(mdtime[:], mv_i[k,:], color='lightcoral')
#			ax[1].plot(dtime[:], v[k,:], color='turquoise')
#			#ax[1].set_xlim((tmin, tmax))
#			ax[1].set_xlim((tmin, tmin+dt.timedelta(days=7)))
#			ax[1].set_ylabel("v [m/s]")
#			ax[0].set_title("Depth %f" %dep[k])
#			fig.savefig(ts + "/" + adcp_files[i][:-3] + "_" + str(-dep[k]) + "_time_series.png")
#			# plot model & observation spectra
#			fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(8,12))
#			ax[0].loglog(mufreq, muspec, color='lightcoral', label="Model")
#			ax[0].loglog(oufreq, ouspec, 'turquoise', label="ADCP")
#			ax[0].legend(loc='best')
#			ax[0].set_ylabel("u spectral energy [$(m/s)^2/cph$]")
#			ax[1].loglog(mvfreq, mvspec, color='lightcoral')
#			ax[1].loglog(ovfreq, ovspec, 'turquoise')
#			ax[1].set_xlabel("frequency [$cph$]")	
#			ax[1].set_xlim((10**-4, 0.5))
#			ax[1].set_ylabel("v spectral energy [$(m/s)^2/cph$]")
#			ax[0].set_title("Depth %f" %dep[k])
#			fig.savefig(spec + "/" + adcp_files[i][:-3] + "_" + str(-dep[k]) + "_spectra.png")
#			# plot scatter of model & observations
#			fig, ax = plt.subplots()
#			ax.scatter(mu_i[k,:], mv_i[k,:], s=0.5, alpha=0.5, color='lightcoral', label='Model')
#			ax.scatter(u[k,:], v[k,:], s=0.5, alpha=0.5, color='turquoise', label='ADCP')
#			lgnd = ax.legend()
#			lgnd.legendHandles[0]._sizes = [5]
#			lgnd.legendHandles[1]._sizes = [5]		
#			ax.set_xlabel("u")
#			ax.set_ylabel("v")
#			ax.set_title("Depth %f" %dep[k])
#			fig.savefig(scat + "/" + adcp_files[i][:-3] + "_" + str(-dep[k]) + "_scatter.png")
	plt.close("all")