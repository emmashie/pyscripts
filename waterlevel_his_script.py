import numpy as np
import matplotlib.pyplot as plt
import waterlevel_validation as wv
from matplotlib.dates import num2date, date2num
import analysis as an
import datetime as dt
from stompy.spatial import proj_utils
import os
import netCDF4 as nc
import xarray as xr
from stompy.io.local import noaa_coops
from stompy import utils

#hisfile = "/home/emma/sfb_dfm_setup/r14/DFM_OUTPUT_r14/his_files/r14_0000*.nc"

# path = "/opt/data/delft/sfb_dfm_v2/runs/wy2013a/DFM_OUTPUT_wy2013a/"
# hisfile = path + "wy2013a_0000_20120801_000000_his.nc"

# RH 2017-11-27: update to run with "better" Delta flows
path = "/opt/data/delft/sfb_dfm_v2/runs/wy2013b/DFM_OUTPUT_wy2013b/"
hisfile = path + "wy2013b_0000_20120801_000000_his.nc"


savepath = path + "/validation_plots/waterlevel_validation_plots/"
metpath = path + "validation_metrics/"
met = "waterlevel_metrics"

if not os.path.exists(savepath):
	os.makedirs(savepath)
if not os.path.exists(metpath):
        os.makedirs(metpath)

ll_to_utm = proj_utils.mapper('WGS84','EPSG:26910')
utm_to_ll = proj_utils.mapper('EPSG:26910','WGS84')

mdat = nc.MFDataset(hisfile)
# pull out coordinates of model stations
xcoor = mdat.variables["station_x_coordinate"][:]
ycoor = mdat.variables["station_y_coordinate"][:]
# convert utm coordinates to lat lon
if xcoor.ndim==2:
	# seems that some DFM versions add a time axis to these.
	# trust that time is the first dimension, and slice it off.
	xcoor=xcoor[0,:]
	ycoor=ycoor[0,:]
mll = utm_to_ll(np.c_[xcoor,ycoor]).T

##         
## Simple caching for tide data to avoid re-fetching too much.
cache_dir=os.path.join(path,'validation_metrics',"cache")
os.path.exists(cache_dir) or os.makedirs(cache_dir)

def noaa_cached(noaa_station,product,start_date,end_date,*a,**k):
        start_dt=utils.to_datetime(start_date)
        end_dt = utils.to_datetime(end_date)
        
        cache_fn=os.path.join(cache_dir,
                              "%s-%s-%s-%s.nc"%(noaa_station,product,
                                                start_dt.strftime("%Y%m%d"),
                                                end_dt.strftime("%Y%m%d")))
        if not os.path.exists(cache_fn):
                dat=noaa_coops.coops_dataset_product(station=noaa_station,
                                                     product=product,
                                                     start_date=start_date,
                                                     end_date=end_date,
                                                     days_per_request=31)
                dat.to_netcdf(cache_fn)
        # even if newly fetched, read from disk to avoid non-reproducible behavior.
        return xr.open_dataset(cache_fn)
        
##

f = open(metpath + met, "w")
f.write("\\begin{center} \n")
f.write("\\begin{adjustbox}{width=1\\textwidth} \n")
f.write("\\begin{tabular}{| l | r | r | r | r | r | r |} \n")
f.write("\\hline \n")
f.write("Name             & Skill   &  Bias (m) & \(r^2\) & RMSE (m) & Lag (min) & Amp. factor\\\ \\hline \n")

def plot_noaa_comparison(noaa_station,title,base_name,datum=''):
        dat =  noaa_cached(noaa_station=noaa_station, product="water_level",
                           start_date=np.datetime64("2012-08-01"),
                           end_date=np.datetime64("2013-09-01"),
                           days_per_request=31)
        lon = dat["lon"].values[0,0]
        lat = dat["lat"].values[0,0]
        time = dat["time"]
        # faster date conversion:
        times=utils.to_dnum(dat.time.values)
        waterlevel = dat["water_level"][0,:]

        dist = np.sqrt((mll[0,:]-lon)**2 +  (mll[1,:]-lat)**2)
        rec = np.argmin(dist) # np.where(dist == np.min(dist))[0]
        mtime, mwaterlevel = wv.load_model(hisfile, rec=rec)
        mtimes = date2num(mtime)

        # Adjust to relative at this point so that model skill doesn't
        # punish us for the arbitrary offset
        if datum=='relative':
                waterlevel = waterlevel-waterlevel.mean()
                mwaterlevel = mwaterlevel-mwaterlevel.mean()
        
        mwaterleveli = np.interp(times, mtimes, np.asarray(mwaterlevel),
                                 left=np.nan,right=np.nan)
        valid=np.isfinite(mwaterleveli+waterlevel.values)

        
        ms, bias, r2, rms, lag, amp = an.model_metrics(times[valid], mwaterleveli[valid],
                                                       times[valid], waterlevel.values[valid])
        # slightly better to find lag on the original data - okay, makes almost no difference.
        lag2 = utils.find_lag(mtimes,mwaterlevel,
                              times, waterlevel)
        
        if datum!='relative':
                bias="%0.3f"%bias
        else:
                bias="--"
        # since times are datenums, lag is in decimal days.
        if f:
                # Note - the spacing here is lined up with the spacing for the header row.
                # not essential, but easier to read in the latex source.
                f.write("%-16s & %7.3f & %9s & %7.3f &  %7.3f &    %6.1f &   %6.2f   \\\ \\hline \n" % (title,
                                                                                                   ms, bias, r2, rms,
                                                                                                   24*60*lag2,
                                                                                                   amp))
        
        # plotting
        fig, ax = plt.subplots(figsize=(6,4))
        ax.plot(time, waterlevel, color='cornflowerblue',lw=1.5)
        # Not as pretty, but maybe more legible
        ax.plot(mtime, mwaterlevel, color='k',lw=0.8) # color='turquoise', alpha=0.75)
        ax.set_title("%s (%s)"%(title,noaa_station))
        ax.legend(["Observed","Model"], loc='best')
        if datum=='relative':
                ax.set_ylabel("Water level anomaly (m)")
        else:
                ax.set_ylabel("Water level (m %s)"%datum)
                
        ax.set_xlim([dt.date(2012,10,1), dt.date(2012,10,7)])
        fig.subplots_adjust(left=0.17) # text was a bit cramped.
        fig.autofmt_xdate()
        fig.savefig(savepath + "%s.png"%base_name)
        fig.savefig(savepath + "%s.pdf"%base_name)

plot_noaa_comparison(noaa_station="9414290",
                     title="San Francisco",
                     base_name="SanFrancisco",
                     datum='NAVD88')

plot_noaa_comparison(noaa_station="9415020",
                     title="Point Reyes",
                     base_name="PointReyes",
                     datum='NAVD88')

plot_noaa_comparison(noaa_station="9414863",
                     title="Richmond",
                     base_name="Richmond",
                     datum='relative')

plot_noaa_comparison(noaa_station="9414750",
                     title="Alameda",
                     base_name="Alameda",
                     datum="NAVD88")

plot_noaa_comparison(noaa_station="9414523",
                     title="Redwood City",
                     base_name="RedwoodCity",
                     datum="relative")

plot_noaa_comparison(noaa_station="9415144",
                     title="Port Chicago",
                     base_name="PortChicago",
                     datum="NAVD88")



# f.write("\\hline \n") # repeats
f.write("\\end{tabular} \n")
f.write("\\end{adjustbox} \n")
f.write("\\end{center} \n")
f.close()
f=None

plt.close('all')

#fig, ax = plt.subplots(nrows=4, sharex=True, figsize=(10,10))
#ax[0].plot(time_1, zeta_1-np.mean(zeta_1), color='cornflowerblue')
#ax[0].plot(t_1, waterlevel_1, color='turquoise', alpha=0.75)
#ax[0].plot(time_1, waterlevel_i_1 - (zeta_1-np.mean(zeta_1)), color='lightcoral')
#ax[0].set_title("9414290")
#ax[0].legend(["obs","model", "model-obs"], loc='best')

#ax[1].plot(time_2, zeta_2-np.mean(zeta_2), color='cornflowerblue')
#ax[1].plot(t_2, waterlevel_2, color='turquoise', alpha=0.75)
#ax[1].plot(time_2, waterlevel_i_2 - (zeta_2-np.mean(zeta_2)), color='lightcoral')
#ax[1].set_title("9415020")

#ax[2].plot(time_3, zeta_3-np.mean(zeta_3), color='cornflowerblue')
#ax[2].plot(t_3, waterlevel_3, color='turquoise', alpha=0.75)
#ax[2].plot(time_3, waterlevel_i_3 - (zeta_3-np.mean(zeta_3)), color='lightcoral')
#ax[2].set_title("9414863")

#ax[3].plot(time_4, zeta_4-np.mean(zeta_4), color='cornflowerblue')
#ax[3].plot(t_4, waterlevel_4, color='turquoise', alpha=0.75)
#ax[3].plot(time_4, waterlevel_i_4 - (zeta_4-np.mean(zeta_4)), color='lightcoral')
#ax[3].set_title("9414750")
#fig.savefig(savepath + "waterlevel_timeseries.png")

#fig, ax = plt.subplots(nrows=4, sharex=True, figsize=(10,10))
#ax[0].loglog(ofreq_1, ospec_1, color='cornflowerblue')
#ax[0].loglog(mfreq_1, mspec_1, color='turquoise')
#ax[0].legend(["obs","model"], loc='best')
#ax[0].set_title("9414290")

#ax[1].loglog(ofreq_2, ospec_2, color='cornflowerblue')
#ax[1].loglog(mfreq_2, mspec_2, color='turquoise')
#ax[1].set_ylabel("spectral energy [$m^2 / cph$]")
#ax[1].set_title("9415020")

#ax[2].loglog(ofreq_3, ospec_3, color='cornflowerblue')
#ax[2].loglog(mfreq_3, mspec_3, color='turquoise')
#ax[2].set_title("9414863")

#ax[3].loglog(ofreq_4, ospec_4, color='cornflowerblue')
#ax[3].loglog(mfreq_4, mspec_4, color='turquoise')
#ax[3].set_xlabel("frequency [$cph$]")
#ax[3].set_title("9414750")
#ax[3].set_xlim((10**-4, 0.5))
#fig.savefig(savepath + "waterlevel_spectra.png")

