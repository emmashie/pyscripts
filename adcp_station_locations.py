import numpy as np
import netCDF4 as nc

path = "/opt/data/noaa/ports/"

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

f = open(path + "st_latlon.txt", "w")
f.write("file_name; station_name; lat; lon \n")
for i in range(len(adcp_files)):
	dat = nc.Dataset(path + adcp_files[i])
	lat = dat["latitude"][:]
	lon = dat["longitude"][:]
	name = dat.station_name
	f.write("%s; %s; %f; %f \n" % (str(adcp_files[i]), str(name), float(lat), float(lon)))
f.close()

