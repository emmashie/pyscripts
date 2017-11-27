from __future__ import print_function

import numpy as np
import netCDF4 as nc
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib.dates import date2num
from stompy.utils import model_skill
from stompy.utils import rotate_to_principal,principal_theta, rot
import analysis as an
from stompy.spatial import proj_utils
import os

## 
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
# output_path = "/opt/data/delft/sfb_dfm_v2/runs/wy2013a/DFM_OUTPUT_wy2013a/"	
# model_files = output_path + "wy2013a_0000_20120801_000000_his.nc"
# 2017-11-27: plotting new run with improved (?) Delta flows
output_path = "/opt/data/delft/sfb_dfm_v2/runs/wy2013b/DFM_OUTPUT_wy2013b/"
model_files = output_path + "wy2013b_0000_20120801_000000_his.nc"

mdat = nc.MFDataset(model_files)

# pull out coordinates of model stations
xcoor = mdat.variables["station_x_coordinate"][:]
ycoor = mdat.variables["station_y_coordinate"][:]
if xcoor.ndim==2:
	# seems that some DFM versions add a time axis to these.
	# trust that time is the first dimension, and slice it off.
	xcoor=xcoor[0,:]
	ycoor=ycoor[0,:]
# convert utm coordinates to lat lon 
mll = utm_to_ll(np.c_[xcoor,ycoor]).T

##

# one-off prep work for ADCP plots - this stuff is slow, so do it outside
# the loop
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

##

# get model station indices for closest station to observation 
llind = np.zeros(len(adcp_files))
for i in range(len(llind)):
	print("Processing %s"%adcp_files[i])
	dir = str(adcp_files[i][:-3])
	dat = nc.Dataset("/opt/data/noaa/ports/" + adcp_files[i])
	dist = np.sqrt((mll[0,:]-dat.variables["longitude"][:])**2 +  (mll[1,:]-dat.variables["latitude"][:])**2)
	llind[i] = np.argmin(dist) # np.where(dist == np.min(dist))[0]
	# define observation variables
	time = dat.variables["time"][:]
	u = dat.variables["u"][:]
	v = dat.variables["v"][:]
	ubar = dat.variables["u_davg"][:]
	vbar = dat.variables["v_davg"][:]
	dep = -dat.variables["depth"][:]
	
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
		try:
			# Get NOAA reported flood direction
			flood_dir=float(dat['flood_dir'][:])
			# convert to "math" convention, and radians
			flood_dir=(90-flood_dir)*np.pi/180.
		except Exception:
			print("Failed to get flood dir?!")
			flood_dir=0.0 # east

		mvec=np.asarray([mubar_i, mvbar_i]).T
		mtheta=principal_theta(mvec,positive=flood_dir)
		muvbar = rot(-mtheta,mvec).T
		vec=np.asarray([ubar, vbar]).T
		# specifying positive=mtheta is supposed to help resolve
		# 180 degree ambiguity
		theta=principal_theta(vec,positive=mtheta)
		uvbar = rot(-theta,vec).T

		obs_color='green'
		mod_color='lightslategray'
		
		if 1: # plotting up model & observation time series
			fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(8,7))
			ax[0].plot(mdtime[:], mubar[:,llind[i]], color=mod_color, label='Model')
			ax[0].plot(dtime[:], ubar[:], '--', color=obs_color, label='ADCP')
			#ax[0].plot(dtime[:], muvbar[0,:], color='lightcoral', label='Model')
			#ax[0].plot(dtime[:], uvbar[0,:], color='turquoise', label='ADCP')
			ax[0].legend()
			ax[0].set_title(dir[:-5])
			ax[0].set_ylabel("Eastward Velocity [m/s]")
			ax[1].plot(mdtime[:], mvbar[:,llind[i]], color=mod_color)
			ax[1].plot(dtime[:], vbar[:], '--', color=obs_color)
			#ax[1].plot(dtime[:], muvbar[1,:], color='lightcoral', label='Model')
			#ax[1].plot(dtime[:], uvbar[1,:], color='turquoise', label='ADCP')
			#ax[1].set_xlim((tmin, tmax))
			ax[1].set_xlim((tmin, tmin+dt.timedelta(days=7)))
			ax[1].set_ylabel("Northward Velocity [m/s]")
			fig.autofmt_xdate(rotation=45)
			fig.savefig(ts + "/" + adcp_files[i][:-3] + "_time_series.png")
			fig.savefig(ts + "/" + adcp_files[i][:-3] + "_time_series.pdf", format='pdf')
                
		if 0: # plot model & observation spectra
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
			fig.savefig(spec + "/" + adcp_files[i][:-3] + "_spectra", format='pdf')
                        
		if 1: # plot scatter of model & observations
			fig, ax = plt.subplots(figsize=(4,4))
			#ax.scatter(mubar[:,llind[i]], mvbar[:,llind[i]], s=0.5, alpha=0.5, color='lightcoral', label='Model')
			#ax.scatter(ubar, vbar, s=0.5, alpha=0.5, color='turquoise', label='ADCP')
			mn = np.min([np.nanmin(muvbar[0,:]), np.nanmin(uvbar[0,:])])
			mx = np.max([np.nanmax(muvbar[0,:]), np.nanmax(uvbar[0,:])])
			lin = np.linspace(mn,mx)
                        # Adjust colors for extra contrast 
			ax.plot(lin,lin, color="k",lw=0.8)
			ind = np.where(muvbar[0,:]!=0)
			ax.scatter(muvbar[0,(muvbar[0,:]!=0) & (uvbar[0,:]!=0)],
                                   uvbar[0,(muvbar[0,:]!=0) & (uvbar[0,:]!=0)],
                                   s=1.25, color='darkgreen')
                        # this causes the axes to shrink as need to maintain a 1:1
                        # aspect ratio
			ax.axis('scaled') 
			ax.set_xlim((mn,mx))
			ax.set_ylim((mn,mx))
                        
			#lgnd = ax.legend()
			#lgnd.legendHandles[0]._sizes = [5]
			#lgnd.legendHandles[1]._sizes = [5]
			#ax.set_xlabel("u")
			#ax.set_ylabel("v")
			ax.set_xlabel("Model [m/s]")
			ax.set_ylabel("Obs [m/s]")
			ax.set_title(dir[:-5])

			if 1: # add a pair of arrows to indicate the positive velocity direction
				xy=(0.7,0.25)
				for a_theta,label,length,col in [(mtheta,'Mod',0.2,mod_color),
								 (theta,'Obs',0.1,obs_color)]:
					xy_tip=(xy[0] + length*np.cos(a_theta),
						xy[1] + length*np.sin(a_theta))

					ax.annotate("",
						    xy=xy_tip,
						    xycoords='axes fraction',
						    xytext=xy,
						    textcoords='axes fraction',
						    arrowprops=dict(arrowstyle="->",connectionstyle="arc3",color=col),
					)
					ax.text( xy_tip[0],xy_tip[1],
						 label,transform=ax.transAxes,color=col,
						 rotation= (a_theta*180/np.pi +90)%180. - 90)


			
			fig.subplots_adjust(left=0.16) # give y label a bit of space
			fig.savefig(scat + "/" + adcp_files[i][:-3] + "_scatter.png")
			fig.savefig(scat + "/" + adcp_files[i][:-3] + "_scatter.pdf", format='pdf')
                
		# ----- VERTICAL LEVELS ------
		# interpolate model to observation depths
                # mu_i = np.zeros((len(dep), len(mdtime)))
                # mv_i = np.zeros((len(dep), len(mdtime)))
                # for j in range(len(mdtime)):
                # 	mu_i[:,j] = np.interp(dep, mdep[j,llind[i],:], mu[j,llind[i],:])
                # 	mv_i[:,j] = np.interp(dep, mdep[j,llind[i],:], mv[j,llind[i],:])
                # # interpolate model to observation times
                # mu_ii = np.zeros((len(dep), len(dtime)))
                # mv_ii = np.zeros((len(dep), len(dtime)))
                # for k in range(len(dep)):
                # 	mu_ii[k,:] = np.interp(dtimes, mtimes, mu_i[k,:])
                # 	mv_ii[k,:] = np.interp(dtimes, mtimes, mv_i[k,:])
                # 	# check for nans in observations
                # 	dumu = u[k,:]
                # 	unan = dumu[~np.isnan(dumu)]
                # 	utime = time[~np.isnan(dumu)]
                # 	dumv = v[k,:]
                # 	vnan = dumv[~np.isnan(dumv)]
                # 	vtime = time[~np.isnan(dumv)]
                # 	# computing observation spectra
                # 	oufreq, ouspec = an.band_avg(utime, unan, dt=(time[1]-time[0])/3600)
                # 	ovfreq, ovspec = an.band_avg(vtime, vnan, dt=(time[1]-time[0])/3600) 
                # 	# computing model spectra
                # 	mufreq, muspec = an.band_avg(mtimes, mu_i[k,:])
                # 	mvfreq, mvspec = an.band_avg(mtimes, mv_i[k,:])
                # 	# plotting up model & observation time series	
                # 	fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(12,6))
                # 	ax[0].plot(mdtime[:], mu_i[k,:], color='lightcoral', label='Model')
                # 	ax[0].plot(dtime[:], u[k,:], color='turquoise', label='ADCP')
                # 	ax[0].legend()
                # 	ax[0].set_ylabel("u [m/s]")
                # 	ax[1].plot(mdtime[:], mv_i[k,:], color='lightcoral')
                # 	ax[1].plot(dtime[:], v[k,:], color='turquoise')
                # 	#ax[1].set_xlim((tmin, tmax))
                # 	ax[1].set_xlim((tmin, tmin+dt.timedelta(days=7)))
                # 	ax[1].set_ylabel("v [m/s]")
                # 	ax[0].set_title("Depth %f" %dep[k])
                # 	fig.savefig(ts + "/" + adcp_files[i][:-3] + "_" + str(-dep[k]) + "_time_series.png")
                # 	# plot model & observation spectra
                # 	fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(8,12))
                # 	ax[0].loglog(mufreq, muspec, color='lightcoral', label="Model")
                # 	ax[0].loglog(oufreq, ouspec, 'turquoise', label="ADCP")
                # 	ax[0].legend(loc='best')
                # 	ax[0].set_ylabel("u spectral energy [$(m/s)^2/cph$]")
                # 	ax[1].loglog(mvfreq, mvspec, color='lightcoral')
                # 	ax[1].loglog(ovfreq, ovspec, 'turquoise')
                # 	ax[1].set_xlabel("frequency [$cph$]")	
                # 	ax[1].set_xlim((10**-4, 0.5))
                # 	ax[1].set_ylabel("v spectral energy [$(m/s)^2/cph$]")
                # 	ax[0].set_title("Depth %f" %dep[k])
                # 	fig.savefig(spec + "/" + adcp_files[i][:-3] + "_" + str(-dep[k]) + "_spectra.png")
                # 	# plot scatter of model & observations
                # 	fig, ax = plt.subplots()
                # 	ax.scatter(mu_i[k,:], mv_i[k,:], s=0.5, alpha=0.5, color='lightcoral', label='Model')
                # 	ax.scatter(u[k,:], v[k,:], s=0.5, alpha=0.5, color='turquoise', label='ADCP')
                # 	lgnd = ax.legend()
                # 	lgnd.legendHandles[0]._sizes = [5]
                # 	lgnd.legendHandles[1]._sizes = [5]		
                # 	ax.set_xlabel("u")
                # 	ax.set_ylabel("v")
                # 	ax.set_title("Depth %f" %dep[k])
                # 	fig.savefig(scat + "/" + adcp_files[i][:-3] + "_" + str(-dep[k]) + "_scatter.png")
	plt.close("all")

