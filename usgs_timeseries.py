"""
Validate against time series of salinity from USGS sites.

Data sources:
 https://waterdata.usgs.gov/nwis/uv?site_no=11162765
 San Mateo Bridge
 Specific conductance at two elevations.
   (uS/cm at 25degC)
 Appears to cover 2007 to present.

 https://waterdata.usgs.gov/nwis/uv?site_no=373025122065901 
 Old Dumbarton Bridge near Newark
 Just has a pressure gage
 Appears to cover 2010 through present.
 On NWIS mapper, this is the eastern station of the two on Dumbarton Br

 https://waterdata.usgs.gov/nwis/inventory?agency_code=USGS&site_no=373015122071000
 Dumbarton Bridge
 This has temp, cond., turbidity, depth, but only turbidity is available before
 Oct 2013, so overall not useful.

 https://waterdata.usgs.gov/nwis/inventory?agency_code=USGS&site_no=374811122235001
 Pier 17, San Francisco 
 Dec 2013 is earliest data.

 https://waterdata.usgs.gov/nwis/uv?site_no=374938122251801
 Alcatraz Island
 Has salinity, spec. cond., includes all of wy2013.

 https://waterdata.usgs.gov/nwis/uv?site_no=375607122264701
 Richmond-San Rafael Bridge
 Two elevations of temp, cond, 2007 to present.

"""
import sys
import os
if 'DISPLAY' not in os.environ:
    import matplotlib
    matplotlib.use('Agg')
    
import numpy as np
import matplotlib.pyplot as plt
from stompy.io.local import usgs_nwis
import stompy.model.delft.io as dio
import seawater
from stompy import utils
import os
import xarray as xr
from scipy.ndimage.filters import percentile_filter
from stompy.spatial import proj_utils
from stompy import filters

##
ll2utm=proj_utils.mapper('WGS84','EPSG:26910')

##

run_name="wy2014"
begindate = "20130801"
path = "/hpcvol1/emma/sfb_dfm/runs/%s/DFM_OUTPUT_%s/"%(run_name,run_name)
hisfile = os.path.join(path, "%s_0000_%s_000000_his.nc"%(run_name,begindate))
mdufile = os.path.join(path,"..","%s.mdu"%run_name)
mdu=dio.MDUFile(mdufile)
t_ref,t_start,t_stop=mdu.time_range()

#t_spunup=np.datetime64("2012-10-01") # clip to "real" period
t_spunup=t_start # show entire simulation

savepath = os.path.join(path + "validation_plots/salinity_time_series/")
metricpath = os.path.join(path + "validation_metrics/")
metric_fn  = os.path.join(metricpath + "salinity_time_series.tex")
os.path.exists(savepath) or os.makedirs(savepath)
os.path.exists(metricpath) or os.makedirs(metricpath)

##
station_locs={
    # elev_mab: upper sensor first on 
    # San Mateo Bridge
    "11162765":dict(lon=-(122 + 14/60. + 59/3600.),
                    lat=37+35/60.+4/3600.,
                    elev_mab=[13.4,3.0]),
    # Alcatraz:
    "374938122251801":dict(lon=-(122+25/60.+18/3600.),
                           lat=37+49/60.+38/3600.,
                           elev_mab=[None] # unknown
    ),
    # Richmond-San Rafael Bridge
    "375607122264701":dict(lon=-(122+26/60.+47/3600.),
                           lat=37+56/60.+7/3600.,
                           elev_mab=[9.1,1.5] ),
    # Alviso Slough
    "11169750":dict(lon=-(121+59/60.+54/3600.),
                    lat=37+26/60.+24/3600.,
                    elev_mab=[None]), 
    # Dumbarton Bridge
    "373015122071000":dict(lon=-(122+7/60.+10/3600.), 
                           lat=37+30/60.+7/3600.,
                           elev_mab=[7.62, 1.22])
}


def usgs_salinity_time_series(station):
    # A little tricky - there are two elevations, which have the same parameter
    # code of 95 for specific conductance, but ts_id's of 14739 and 14741.
    # requesting the parameter once does return an RDB with both in there.

    time_labels=[utils.to_datetime(t).strftime('%Y%m%d')
                 for t in [t_start,t_stop]]
    cache_fn="usgs%s-%s_%s-salinity.nc"%(station,time_labels[0],time_labels[1])

    if not os.path.exists(cache_fn):

        # 95: specific conductance
        # 90860: salinity
        ds=usgs_nwis.nwis_dataset(station,
                                  t_start,t_stop,
                                  products=[95,90860],
                                  days_per_request=20)
        usgs_nwis.add_salinity(ds)
        ds.to_netcdf(cache_fn)
        ds.close()

    ##

    ds=xr.open_dataset(cache_fn)

    ds.attrs['lon']=station_locs[station]['lon']
    ds.attrs['lat']=station_locs[station]['lat']
    ds.salinity.attrs['elev_mab']=station_locs[station]['elev_mab'][0]
    if 'salinity_01' in ds:
        ds.salinity_01.attrs['elev_mab']=station_locs[station]['elev_mab'][1]

    xy=ll2utm( [ds.attrs['lon'],ds.attrs['lat']] )
    ds.attrs['x']=xy[0]
    ds.attrs['y']=xy[1]
    return ds

usgs_salinity_time_series(station="11169750")
##

# Minor tweaks to model output:
his=xr.open_dataset(hisfile)
his_xy=np.c_[his.station_x_coordinate.isel(time=0).values,
             his.station_y_coordinate.isel(time=0).values]
his_dt_s=np.median(np.diff(his.time)) / np.timedelta64(1,'s')

mod_stride=slice(None,None,max(1,int(3600./his_dt_s)))

##
def extract_at_zab(his,field,z_mab,**sel_kw):
    depths=his.waterdepth.isel(**sel_kw)
    mean_depth=depths.mean()
    z_bed=his.zcoordinate_w.isel(time=0,laydimw=0,**sel_kw)
    z_sel=float(z_bed+z_mab)
    his_z=his.zcoordinate_c.isel(**sel_kw).values
    his_values=his[field].isel(**sel_kw).values
    values_at_z=np.array( [ np.interp(z_sel,
                                      his_z[i,:],his_values[i,:])
                            for i in range(len(his.time)) ] )
    return values_at_z

tex_fp=None        

def figure_usgs_salinity_time_series(station,station_name):
    mod_lp_win=usgs_lp_win=40 # 40h lowpass 

    # Gather USGS data:
    ds=usgs_salinity_time_series(station)
    usgs_dt_s=np.median(np.diff(ds.time)) / np.timedelta64(1,'s')
    usgs_stride=slice(None,None,max(1,int(3600./usgs_dt_s)))
    if 'salinity_01' in ds:
        obs_salt_davg=np.c_[ds.salinity.values[usgs_stride],
                            ds.salinity_01.values[usgs_stride]]
        obs_salt_davg=np.nanmean(obs_salt_davg,axis=1)
    else:
        obs_salt_davg=ds.salinity.values[usgs_stride]


    dists= utils.dist( his_xy, [ds.x,ds.y] ) 
    station_idx=np.argmin(dists)
    print("Nearest model station is %.0f m away from observation"%(dists[station_idx]))
    print(station_idx)

    def low_high(d,winsize):
        high=percentile_filter(d,95,winsize)
        low=percentile_filter(d,5,winsize)
        high=filters.lowpass_fir(high,winsize)
        low=filters.lowpass_fir(low,winsize)
        return low,high

    obs_salt_range=low_high(obs_salt_davg,usgs_lp_win)

    mod_salt_davg=his.salinity.isel(stations=station_idx).mean(dim='laydim')
    mod_salt_range=low_high(mod_salt_davg,mod_lp_win)

    # Try picking out a reasonable depth in the model
    surf_label="Surface"
    bed_label="Bed"
    if ds.salinity.attrs['elev_mab'] is not None:
        z_mab=ds.salinity.attrs['elev_mab']
        surf_label="%.1f mab"%z_mab
        mod_salt_surf=extract_at_zab(his,"salinity",z_mab,stations=station_idx)
    else:
        mod_salt_surf=his.salinity.isel(stations=station_idx,laydim=-1)

    if ('salinity_01' in ds) and (ds.salinity_01.attrs['elev_mab'] is not None):
        z_mab=ds.salinity_01.attrs['elev_mab']
        bed_label="%.1f mab"%z_mab
        mod_salt_bed=extract_at_zab(his,"salinity",z_mab,stations=station_idx)
    else:
        mod_salt_bed=his.salinity.isel(stations=station_idx,laydim=0)


    mod_deltaS=mod_salt_bed - mod_salt_surf

    if 'salinity_01' in ds:
        if ds.site_no == '375607122264701':
            ds = ds.rename({"salinity": "salinity_01", "salinity_01": "salinity"})

    if 'salinity_01' in ds:
        obs_deltaS=ds.salinity_01.values - ds.salinity.values
    else:
        obs_deltaS=None
        
    if 1: # plotting time series
        plt.figure(1).clf()
        fig,ax=plt.subplots(num=1)
        fig.set_size_inches([10,4.75],forward=True)

        # These roughly mimic the style of water level plots in the validation report.
        obs_color='cornflowerblue'
        obs_lw=1.5
        mod_color='k'
        mod_lw=0.8

        if 'salinity_01' in ds:
            ax.plot(utils.to_dnum(ds.time)[usgs_stride],
                    filters.lowpass_fir(ds.salinity[usgs_stride],usgs_lp_win),
                    label='Obs. Upper',lw=obs_lw,color=obs_color)

            ax.plot(utils.to_dnum(ds.time)[usgs_stride],
                    filters.lowpass_fir(ds.salinity_01[usgs_stride],usgs_lp_win),
                    label='Obs. Lower',lw=obs_lw,color=obs_color,ls='--')
        else:
            ax.plot(utils.to_dnum(ds.time)[usgs_stride],
                    filters.lowpass_fir(ds.salinity[usgs_stride],usgs_lp_win),
                    label='Obs.',lw=obs_lw,color=obs_color)
            
        ax.plot(utils.to_dnum(his.time),
                filters.lowpass_fir(mod_salt_surf,40),
                label='Model %s'%surf_label,lw=mod_lw,color=mod_color)

        ax.plot(utils.to_dnum(his.time),
                filters.lowpass_fir(mod_salt_bed,40),
                label='Model %s'%bed_label,lw=mod_lw,color=mod_color,ls='--')


        if 1: # is it worth showing tidal variability?

            ax.fill_between(utils.to_dnum(ds.time)[usgs_stride],
                            obs_salt_range[0],obs_salt_range[1],
                            color=obs_color,alpha=0.3,zorder=-1,lw=0)

            ax.fill_between(utils.to_dnum(his.time),
                            mod_salt_range[0],mod_salt_range[1],
                            color='0.3',alpha=0.3,zorder=-1,lw=0)

        ax.set_title(station_name)
        ax.xaxis.axis_date()
        fig.autofmt_xdate()
        ax.set_ylabel('Salinity (ppt)')
        ax.legend(fontsize=10,loc='lower left')

        ax.axis(xmin=utils.to_dnum(t_spunup),
                xmax=utils.to_dnum(t_stop))
        
        fig.tight_layout()

        safe_station=station_name.replace(' ','_')
        fig.savefig(os.path.join(savepath,"%s.png"%safe_station),
                    dpi=100)
        fig.savefig(os.path.join(savepath,"%s.pdf"%safe_station))

    if tex_fp is not None:  # metrics
        target_time_dnum=utils.to_dnum(his.time.values)

        obs_time_dnum=utils.to_dnum(ds.time.values)

        obs_salt_davg_intp=utils.interp_near( target_time_dnum,
                                              obs_time_dnum[usgs_stride], obs_salt_davg,
                                              1.5/24 )
        valid=np.isfinite(mod_salt_davg*obs_salt_davg_intp).values
        valid=( valid
                & (target_time_dnum>=utils.to_dnum(t_spunup))
                & (target_time_dnum<=utils.to_dnum(t_stop)) )
        dnum=target_time_dnum[valid]
        mod_values=mod_salt_davg[valid].values
        obs_values=obs_salt_davg_intp[valid]
        
        bias=np.mean(mod_values - obs_values)
        ms=utils.model_skill(mod_values,obs_values)
        r2=np.corrcoef(mod_values,obs_values)[0,1]
        rmse=utils.rms(mod_values - obs_values)
        tex_fp.write( ("%-16s  " # station name
                       " & %7.3f" # skill
                       " & %11.2f" # bias
                       " & %7.3f" # r2
                       " & %10.2f" # rmse
                       " \\\ \\hline \n")%( station_name,ms,bias,r2,rmse)  )

##


tex_fp=open(metric_fn,"wt")
tex_fp.write("%% output from %s\n"%metric_fn)
tex_fp.write("% Name             & Skill   &  Bias (ppt) & \(r^2\) & RMSE (ppt) \\\ \\hline \n")

# San Mateo Salt:
figure_usgs_salinity_time_series(station="11162765",station_name="San Mateo Bridge")

# 
# Alcatraz:
figure_usgs_salinity_time_series(station="374938122251801",station_name="Alcatraz")

# Richmond/San Rafael Bridge
figure_usgs_salinity_time_series(station="375607122264701",station_name="Richmond Bridge")

figure_usgs_salinity_time_series(station="11169750",station_name="Alviso Slough")

figure_usgs_salinity_time_series(station="373015122071000",station_name="Dumbarton Bridge")


if tex_fp != sys.stdout:
    tex_fp.close()
