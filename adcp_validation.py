from __future__ import print_function

import numpy as np
import netCDF4 as nc
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib.dates import date2num
from stompy.utils import model_skill
from stompy.utils import rotate_to_principal,principal_theta, rot
from pyscripts import analysis as an
from stompy.spatial import proj_utils
import os

## 
ll_to_utm = proj_utils.mapper('WGS84','EPSG:26910')
utm_to_ll = proj_utils.mapper('EPSG:26910','WGS84')

south_adcps = ["SFB1301-2013.nc", "SFB1302-2013.nc", "SFB1304-2013.nc", "SFB1305-2013.nc", 
           "SFB1306-2013.nc", "SFB1307-2013.nc", "SFB1308-2013.nc"]
central_adcps = ["SFB1202-2012.nc", "SFB1203-2012.nc", "SFB1204-2012.nc", "SFB1205-2012.nc", 
         "SFB1206-2012.nc", "SFB1207-2012.nc", "SFB1208-2012.nc", "SFB1209-2012.nc",
         "SFB1210-2012.nc", "SFB1211-2012.nc", "SFB1212-2012.nc", "SFB1213-2012.nc",
         "SFB1214-2012.nc", "SFB1215-2012.nc", "SFB1216-2012.nc", "SFB1217-2012.nc",
         "SFB1218-2012.nc", "SFB1219-2012.nc", "SFB1309-2013.nc", "SFB1310-2013.nc",
         "SFB1311-2013.nc", "SFB1312-2013.nc"]
coastal_adcps = ["SFB1201-2012.nc", "SFB1220-2012.nc", "SFB1221-2012.nc", "SFB1222-2012.nc",
         "SFB1223-2012.nc"]
north_adcps = ["SFB1313-2013.nc", "SFB1314-2013.nc", "SFB1315-2013.nc", "SFB1316-2013.nc",
           "SFB1317-2013.nc", "SFB1318-2013.nc", "SFB1319-2013.nc", "SFB1320-2013.nc",
           "SFB1322-2013.nc", "SFB1323-2013.nc", "SFB1324-2013.nc", "SFB1325-2013.nc",
           "SFB1326-2013.nc", "SFB1327-2013.nc", "SFB1328-2013.nc", "SFB1329-2013.nc",
           "SFB1330-2013.nc", "SFB1331-2013.nc", "SFB1332-2013.nc"]

adcp_files =south_adcps+central_adcps+coastal_adcps+north_adcps

#output_path = "/home/emma/sfb_dfm_setup/r14/DFM_OUTPUT_r14/his_files/"            
#model_files = output_path + "r14_0000_201*.nc"
# output_path = "/opt/data/delft/sfb_dfm_v2/runs/wy2013a/DFM_OUTPUT_wy2013a/"    
# model_files = output_path + "wy2013a_0000_20120801_000000_his.nc"
# 2017-11-27: plotting new run with improved (?) Delta flows
# output_path = "/opt/data/delft/sfb_dfm_v2/runs/wy2013b/DFM_OUTPUT_wy2013b/"
# model_files = output_path + "wy2013b_0000_20120801_000000_his.nc"
run_name="wy2013_temp"
output_path = "/hpcvol1/emma/sfb_dfm/runs/%s/DFM_OUTPUT_%s/"%(run_name,run_name)
model_files = output_path + "%s_0000_20120801_000000_his.nc"%run_name

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

save_metrics=True
rotate_metrics=True # evaluate metrics on the principal and secondary velocities
rotate_series=True # time series show rotated velocities

met_path=os.path.join(output_path,"validation_metrics")
if not os.path.exists(met_path):
    os.makedirs(met_path)
        
# get model station indices for closest station to observation 
for region_name,region_adcps in [ ('south',south_adcps),
                                  ('central',central_adcps),
                                  ('coastal',coastal_adcps),
                                  ('north',north_adcps) ]:
    if save_metrics:
        # open file to write to 
        metrics_fp = open(os.path.join(met_path,"%s_adcp_metrics"%region_name), "wt")
        # write header lines
        metrics_fp.write("\\begin{center} \n")
        metrics_fp.write("\\begin{adjustbox}{width=1\\textwidth} \n")
        metrics_fp.write("\\begin{tabular}{| l | r | r | r | r | r | r | r | r | r | r | r | r |} \n")
        metrics_fp.write("\\hline \n")

        if 1: # split header to two rows:
            metrics_fp.write( '        & \multicolumn{2}{|l|}{Skill} '
                              '& \multicolumn{2}{|l|}{Bias ($m\ s^{-1}$)} '
                              '& \multicolumn{2}{|l|}{\(r^2\)} '
                              '& \multicolumn{2}{|l|}{RMSE ($m\ s^{-1}$)} '
                              '& \multicolumn{2}{|l|}{Lag (min)} '
                              '& \multicolumn{2}{|l|}{Amp. factor} \\\ \hline \n')
            if not rotate_metrics:
                metrics_fp.write('Name    & East & North                '
                                 '& East & North'
                                 '& East & North'
                                 '& East & North'
                                 '& East &  North \\\ \hline\n')
            else:                
                metrics_fp.write('Name   & Pri. & Sec.                '
                                 '& Pri. & Sec. '
                                 '& Pri. & Sec. '
                                 '& Pri. & Sec. '
                                 '& Pri. & Sec. '
                                 '& Pri. & Sec. \\\ \hline\n')
        else:
            metrics_fp.write( ("Name    & Skill (East) & Skill (North) "
                           "& Bias, East ($m s^{-1}$) & Bias, North ($m s^{-1}$) "
                           "& \(r^2\), East & \(r^2\), North "
                           "& RMSE, East ($m s^{-1}$) & RMSE ,North ($m s^{-1}$) "
                           "& Lag, East (min) & Lag, North (min) \\\ \\hline \n") )
            
    for adcp_file in region_adcps:
        print("Processing %s"%adcp_file)
        dir = str(adcp_file[:-3])
        dat = nc.Dataset("/opt/data/noaa/ports/" + adcp_file)
        dist = np.sqrt((mll[0,:]-dat.variables["longitude"][:])**2 +  (mll[1,:]-dat.variables["latitude"][:])**2)
        llind = np.argmin(dist) 
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

        if  tmin >= tmax:
            print("   Not enough data -- skipping")
            continue

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
        mubar_i = np.interp(dtimes, mtimes, mubar[:,llind],left=np.nan,right=np.nan)
        mvbar_i = np.interp(dtimes, mtimes, mvbar[:,llind],left=np.nan,right=np.nan)
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
            if not rotate_series:
                ax[0].plot(mdtime[:], mubar[:,llind], color=mod_color, label='Model')
                ax[0].plot(dtime[:], ubar[:], '--', color=obs_color, label='ADCP')
                
                ax[1].plot(mdtime[:], mvbar[:,llind], color=mod_color)
                ax[1].plot(dtime[:], vbar[:], '--', color=obs_color)
                
                ax[0].set_ylabel("Eastward Velocity [m/s]")
                ax[1].set_ylabel("Northward Velocity [m/s]")
            else:
                rot_muvbar=rot(-mtheta,
                               np.array([mubar[:,llind],
                                         mvbar[:,llind]]).T )
                ax[0].plot(mdtime[:], rot_muvbar[:,0], color=mod_color, label='Model')
                ax[0].plot(dtime[:], uvbar[0,:], '--', color=obs_color, label='ADCP')

                ax[1].plot(mdtime[:], rot_muvbar[:,1], color=mod_color)
                ax[1].plot(dtime[:], uvbar[1,:], '--', color=obs_color)
                
                ax[0].set_ylabel("Principal Velocity [m/s]")
                ax[1].set_ylabel("Secondary Velocity [m/s]")
            
            ax[0].legend(loc='upper right') # specify for consistency
            ax[0].set_title(dir[:-5])

            ax[1].set_xlim((tmin, tmin+dt.timedelta(days=7)))
            fig.autofmt_xdate(rotation=45)
            fig.savefig(ts + "/" + adcp_file[:-3] + "_time_series.png")
            fig.savefig(ts + "/" + adcp_file[:-3] + "_time_series.pdf", format='pdf')

        if 0: # plot model & observation spectra
            # computing observation spectra
            oufreq, ouspec = an.band_avg(time, ubar, dt=(time[1]-time[0])/3600)
            ovfreq, ovspec = an.band_avg(time, vbar, dt=(time[1]-time[0])/3600) 
            #oufreq, ouspec = an.band_avg(time, uvbar[0,:], dt=(time[1]-time[0])/3600)
            #ovfreq, ovspec = an.band_avg(time, uvbar[1,:], dt=(time[1]-time[0])/3600) 
            # computing model spectra
            mufreq, muspec = an.band_avg(mtimes, mubar[:,llind])
            mvfreq, mvspec = an.band_avg(mtimes, mvbar[:,llind])
            #mufreq, muspec = an.band_avg(time, muvbar[0,:], dt=(time[1]-time[0])/3600)
            #mvfreq, mvspec = an.band_avg(time, muvbar[1,:], dt=(time[1]-time[0])/3600)

            fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(5,8))
            ax[0].loglog(mufreq, muspec, color='lightcoral', label='Model')
            ax[0].loglog(oufreq, ouspec, 'turquoise', label='ADCP')
            ax[0].legend(loc='upper right') # specify for consistency
            ax[0].set_ylabel("ubar spectral energy [$(m/s)^2/cph$]")
            ax[0].set_title(dir[:-5])
            ax[1].loglog(mvfreq, mvspec, color='lightcoral')
            ax[1].loglog(ovfreq, ovspec, 'turquoise')
            ax[1].set_xlabel("frequency [$cph$]")    
            ax[1].set_xlim((10**-4, 0.5))
            ax[1].set_ylabel("vbar spectral energy [$(m/s)^2/cph$]")
            fig.savefig(spec + "/" + adcp_file[:-3] + "_spectra.png")
            fig.savefig(spec + "/" + adcp_file[:-3] + "_spectra", format='pdf')

        if 1: # plot scatter of model & observations
            plt.figure(3).clf()
            fig, ax = plt.subplots(figsize=(4,4),num=3)

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

            ax.set_xlabel("Model [m/s]")
            ax.set_ylabel("Obs [m/s]")
            ax.set_title(dir[:-5])

            mid_y=0.5*(np.sin(mtheta)+np.sin(theta))

            if 1: # add a pair of arrows to indicate the positive velocity direction
                length=0.17
                xy=(1-length,length)
                for a_theta,label,lw,col in [(mtheta,'Mod',0.8,mod_color),
                                 (theta,'Obs',1.4,obs_color)]:
                    xy_tip=(xy[0] + length*np.cos(a_theta),
                            xy[1] + length*np.sin(a_theta))
                    # Draw the arrow:
                    ax.annotate("",
                            xy=xy_tip,
                            xycoords='axes fraction',
                            xytext=xy,
                            textcoords='axes fraction',
                            arrowprops=dict(arrowstyle="->",connectionstyle="arc3",color=col),
                    )
                    xy_label=(xy[0]+0.65*length*np.cos(a_theta),
                              xy[1]+0.65*length*np.sin(a_theta))
                    if np.sin(a_theta) > mid_y:
                        # And the label separately
                        ax.text( xy_label[0],xy_label[1],
                                 "%s\n "%label,transform=ax.transAxes,color=col,
                                 va='center',ha='center',
                                 rotation= (a_theta*180/np.pi +90)%180. - 90)
                    else:
                        # And the label separately
                        ax.text(  xy_label[0],xy_label[1],
                                  "\n%s"%label,transform=ax.transAxes,color=col,
                                  va='center',ha='center',
                                  rotation= (a_theta*180/np.pi +90)%180. - 90)
                ax.fill([xy[0]-length,xy[0]+length,xy[0]+length,xy[0]-length],
                        [xy[1]-length,xy[1]-length,xy[1]+length,xy[1]+length],
                        color='w',alpha=0.5,transform=ax.transAxes)

            fig.subplots_adjust(left=0.19) # give y label a bit of space
            # assert False #DBG
            fig.savefig(scat + "/" + adcp_file[:-3] + "_scatter.png")
            fig.savefig(scat + "/" + adcp_file[:-3] + "_scatter.pdf", format='pdf')

        if save_metrics:
            sel=(time >= tmin.timestamp()) & (time <= tmax.timestamp())
            if not rotate_metrics:
                model_u=mubar_i
                model_v=mvbar_i
                obs_u=ubar
                obs_v=vbar
            else:
                model_u=muvbar[0,:]
                model_v=muvbar[1,:]
                obs_u=uvbar[0,:]
                obs_v=uvbar[1,:]
                                
            msu, biasu, r2u, rmsu, lagu, ampu = an.model_metrics(dtimes[sel],model_u[sel],
                                                                 dtimes[sel],obs_u[sel])
            msv, biasv, r2v, rmsv, lagv, ampv = an.model_metrics(dtimes[sel], model_v[sel],
                                                                 dtimes[sel], obs_v[sel])
                
            metrics_fp.write( ("%s & %6.3f & %6.3f & %6.3f & %6.3f & %6.3f & %6.3f "
                               "& %6.3f & %6.3f & %5.2f & %5.1f & %5.2f & %5.2f \\\ \\hline \n") %
                              (dir[:-5], msu, msv, biasu, biasv, r2u, r2v, rmsu, rmsv,
                               24*60*lagu, 24*60*lagv, ampu, ampv))
                          
    if save_metrics:
        metrics_fp.write("\\end{tabular} \n")
        metrics_fp.write("\\end{adjustbox} \n")
        metrics_fp.write("\\end{center} \n")
        metrics_fp.close()

    plt.close("all")


##
