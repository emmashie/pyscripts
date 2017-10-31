import numpy as np
from oceantools import seawater
import matplotlib.pyplot as plt
import netCDF4 as nc

ind = 25
his = nc.MFDataset("r14_0000_201*.nc")
salt = his["salinity"]
temp = his["temperature"]
z = his["zcoordinate_c"]
rho = seawater.dens(temp[:,ind,:], salt[:,ind,:], -z[:,ind,:])
n2 = seawater.buoyancy(rho, z[:,ind,:], len(rho[:,0]), len(z[0,ind,:]))

