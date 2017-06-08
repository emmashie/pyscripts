import numpy as np
import matplotlib.pyplot as plt
import waterlevel_validation as wv
from matplotlib.dates import num2date, date2num
import analysis as an

noaa9414290 = "/home/emma/validation_data/NOAA_WaterLevel/NOAA_9414290.csv"
noaa9415020 = "/home/emma/validation_data/NOAA_WaterLevel/NOAA_9415020.csv"
noaa9414863 = "/home/emma/validation_data/NOAA_WaterLevel/NOAA_9414863.csv"
noaa9414750 = "/home/emma/validation_data/NOAA_WaterLevel/NOAA_9414750.csv"
noaa9414523 = "/home/emma/validation_data/NOAA_WaterLevel/NOAA_9414523.csv"
hisfile = "/home/emma/sfb_dfm_setup/r14/DFM_OUTPUT_r14/his_files/r14_0000*.nc"

##### station 9414290
time_1, zeta_1 = wv.load_NOAA(noaa9414290, header=2, usecols=[8,10])
time_ts_1 = date2num(time_1)
ofreq_1, ospec_1 = an.band_avg(time_ts_1, zeta_1 - np.mean(zeta_1), dt=0.1)
t_1, waterlevel_1 = wv.load_model(hisfile, rec=56)
t_ts_1 = date2num(t_1)
waterlevel_i_1 = np.interp(time_ts_1, t_ts_1, waterlevel_1)
mfreq_1, mspec_1 = an.band_avg(t_ts_1, waterlevel_1)

##### station 9415020
time_2, zeta_2 = wv.load_NOAA(noaa9415020, header=2, usecols=[8,10])
time_ts_2 = date2num(time_2)
ofreq_2, ospec_2 = an.band_avg(time_ts_2, zeta_2 - np.mean(zeta_2), dt=0.1)
t_2, waterlevel_2 = wv.load_model(hisfile, rec=40)
t_ts_2 = date2num(t_2)
waterlevel_i_2 = np.interp(time_ts_2, t_ts_2, waterlevel_2)
mfreq_2, mspec_2 = an.band_avg(t_ts_2, waterlevel_2)

##### station 9414863
time_3, zeta_3 = wv.load_NOAA(noaa9414863, header=2, usecols=[8,10])
time_ts_3 = date2num(time_3)
ofreq_3, ospec_3 = an.band_avg(time_ts_3, zeta_3 - np.mean(zeta_3), dt=0.1)
t_3, waterlevel_3 = wv.load_model(hisfile, rec=149)
t_ts_3 = date2num(t_3)
waterlevel_i_3 = np.interp(time_ts_3, t_ts_3, waterlevel_3)
mfreq_3, mspec_3 = an.band_avg(t_ts_3, waterlevel_3)

##### station 9414750
time_4, zeta_4 = wv.load_NOAA(noaa9414750, header=2, usecols=[8,10])
time_ts_4 = date2num(time_4)
ofreq_4, ospec_4 = an.band_avg(time_ts_4, zeta_4 - np.mean(zeta_4), dt=0.1)
t_4, waterlevel_4 = wv.load_model(hisfile, rec=38)
t_ts_4 = date2num(t_4)
waterlevel_i_4 = np.interp(time_ts_4, t_ts_4, waterlevel_4)
mfreq_4, mspec_4 = an.band_avg(t_ts_4, waterlevel_4)

##### station 9414523
time_5, zeta_5 = wv.load_NOAA(noaa9414523, header=2, usecols=[8,10])
time_ts_5 = date2num(time_5)
ofreq_5, ospec_5 = an.band_avg(time_ts_5, zeta_5 - np.mean(zeta_5), dt=0.1)
t_5, waterlevel_5 = wv.load_model(hisfile, rec=37)
t_ts_5 = date2num(t_5)
waterlevel_i_5 = np.interp(time_ts_5, t_ts_5, waterlevel_5)
mfreq_5, mspec_5 = an.band_avg(t_ts_5, waterlevel_5)


fig, ax = plt.subplots(nrows=4, sharex=True, figsize=(10,10))
ax[0].plot(time_1, zeta_1-np.mean(zeta_1), color='cornflowerblue')
ax[0].plot(t_1, waterlevel_1, color='turquoise', alpha=0.75)
#ax[0].plot(time_1, waterlevel_i_1 - (zeta_1-np.mean(zeta_1)), color='lightcoral')
ax[0].set_title("9414290")
ax[0].legend(["obs","model", "model-obs"], loc='best')

ax[1].plot(time_2, zeta_2-np.mean(zeta_2), color='cornflowerblue')
ax[1].plot(t_2, waterlevel_2, color='turquoise', alpha=0.75)
#ax[1].plot(time_2, waterlevel_i_2 - (zeta_2-np.mean(zeta_2)), color='lightcoral')
ax[1].set_title("9415020")

ax[2].plot(time_3, zeta_3-np.mean(zeta_3), color='cornflowerblue')
ax[2].plot(t_3, waterlevel_3, color='turquoise', alpha=0.75)
#ax[2].plot(time_3, waterlevel_i_3 - (zeta_3-np.mean(zeta_3)), color='lightcoral')
ax[2].set_title("9414863")

ax[3].plot(time_4, zeta_4-np.mean(zeta_4), color='cornflowerblue')
ax[3].plot(t_4, waterlevel_4, color='turquoise', alpha=0.75)
#ax[3].plot(time_4, waterlevel_i_4 - (zeta_4-np.mean(zeta_4)), color='lightcoral')
ax[3].set_title("9414750")
fig.savefig("validation_plots/waterlevel_validation_plots/waterlevel_timeseries.png")

fig, ax = plt.subplots(nrows=4, sharex=True, figsize=(10,10))
ax[0].loglog(ofreq_1, ospec_1, color='cornflowerblue')
ax[0].loglog(mfreq_1, mspec_1, color='turquoise')
ax[0].legend(["obs","model"], loc='best')
ax[0].set_title("9414290")

ax[1].loglog(ofreq_2, ospec_2, color='cornflowerblue')
ax[1].loglog(mfreq_2, mspec_2, color='turquoise')
ax[1].set_ylabel("spectral energy [$m^2 / cph$]")
ax[1].set_title("9415020")

ax[2].loglog(ofreq_3, ospec_3, color='cornflowerblue')
ax[2].loglog(mfreq_3, mspec_3, color='turquoise')
ax[2].set_title("9414863")

ax[3].loglog(ofreq_4, ospec_4, color='cornflowerblue')
ax[3].loglog(mfreq_4, mspec_4, color='turquoise')
ax[3].set_xlabel("frequency [$cph$]")
ax[3].set_title("9414750")
ax[3].set_xlim((10**-4, 0.5))
fig.savefig("validation_plots/waterlevel_validation_plots/waterlevel_spectra.png")

