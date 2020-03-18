import numpy as np
import matplotlib.pyplot as plt
from pyscripts import waterlevel_validation as wv
from matplotlib.dates import num2date, date2num
from pyscripts import analysis as an
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
# RH 2017-12-04: update to attenuated tides and possible evaporation
run_name="wy2014"
path = "/hpcvol1/emma/sfb_dfm/runs/%s/DFM_OUTPUT_%s/"%(run_name,run_name)
hisfile = path + "%s_0000_20130801_000000_his.nc"%run_name


savepath = path + "/validation_plots/waterlevel_validation_plots/"
metpath = path + "validation_metrics/"
met = "waterlevel_metrics"

start_date=np.datetime64("2013-08-01")
end_date=np.datetime64('2014-04-01')
ref_year = 2013

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
# formatting of the table is included but commented out so that minor tweaks
# can be made in the final tex document
f.write("% \\begin{center} \n")
f.write("% \\begin{adjustbox}{width=1\\textwidth} \n")
f.write("% \\begin{tabular}{| l | r | r | r | r | r | r |} \n")
f.write("% \\hline \n")
f.write("% Name             & Skill   &  Bias (m) & \(r^2\) & RMSE (m) & Lag (min) & Amp. factor\\\ \\hline \n")

def plot_noaa_comparison(noaa_station,title,base_name,datum='',start_date=np.datetime64("2016-08-01"),end_date=np.datetime64("2017-09-01")):
        dat =  noaa_cached(noaa_station=noaa_station, product="water_level",
                           start_date=start_date,
                           end_date=end_date,
                           days_per_request=31,
                           ref_year=ref_year)
        lon = dat["lon"].values[0]
        lat = dat["lat"].values[0]
        time = dat["time"]
        # faster date conversion:
        times=utils.to_dnum(dat.time.values)
        waterlevel = dat["water_level"].isel(station=0)

        dist = np.sqrt((mll[0,:]-lon)**2 +  (mll[1,:]-lat)**2)
        rec = np.argmin(dist) # np.where(dist == np.min(dist))[0]
        mtime, mwaterlevel = wv.load_model(hisfile, ref_year=ref_year, rec=rec)
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
        ax.plot(utils.to_dnum(time), waterlevel, color='cornflowerblue',lw=1.5)
        # Not as pretty, but maybe more legible
        ax.plot(mtime, mwaterlevel, color='k',lw=0.8) # color='turquoise', alpha=0.75)
        ax.xaxis.axis_date()
        
        ax.set_title("%s (%s)"%(title,noaa_station))
        ax.legend(["Observed","Model"], loc='best')
        if datum=='relative':
                ax.set_ylabel("Water level anomaly (m)")
        else:
                ax.set_ylabel("Water level (m %s)"%datum)
                
        ax.set_xlim([start_date+np.timedelta64(30*2,'D'), start_date+np.timedelta64(30*2,'D')+np.timedelta64(7,'D')])
        fig.subplots_adjust(left=0.17) # text was a bit cramped.
        fig.autofmt_xdate()
        fig.savefig(savepath + "%s.png"%base_name)
        fig.savefig(savepath + "%s.pdf"%base_name)

        
plot_noaa_comparison(noaa_station="9414290",
                     title="San Francisco",
                     base_name="SanFrancisco",
                     datum='NAVD88',
                     start_date=start_date,
                     end_date=end_date)

plot_noaa_comparison(noaa_station="9415020",
                     title="Point Reyes",
                     base_name="PointReyes",
                     datum='NAVD88',                     
                     start_date=start_date,
                     end_date=end_date)

plot_noaa_comparison(noaa_station="9414863",
                     title="Richmond",
                     base_name="Richmond",
                     datum='relative',
                     start_date=start_date,
                     end_date=end_date)

plot_noaa_comparison(noaa_station="9414750",
                     title="Alameda",
                     base_name="Alameda",
                     datum="NAVD88",
                     start_date=start_date,
                     end_date=end_date)

plot_noaa_comparison(noaa_station="9414523",
                     title="Redwood City",
                     base_name="RedwoodCity",
                     datum="relative",                     
                     start_date=start_date,
                     end_date=end_date)

plot_noaa_comparison(noaa_station="9415144",
                     title="Port Chicago",
                     base_name="PortChicago",
                     datum="NAVD88",                     
                     start_date=start_date,
                     end_date=end_date)

f.write("% \\end{tabular} \n")
f.write("% \\end{adjustbox} \n")
f.write("% \\end{center} \n")
f.close()
f=None

plt.close('all')

