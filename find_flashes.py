from GLD_file_tools import GLD_file_tools
from satellite import Satellite

import datetime
import ephem
import numpy as np
from coordinate_structure import coordinate_structure
from longitude_scaling import longitude_scaling
from ionoAbsorp import ionoAbsorp
#from mpl_toolkits.basemap import Basemap
from matplotlib import pyplot as plt
# Check 
GLD_root  = 'alex/array/home/Vaisala/feed_data/GLD'
NLDN_root = 'alex/array/home/Vaisala/feed_data/NLDN'

sat_TLE  = ["1 40378U 15003C   15293.75287141  .00010129  00000-0  48835-3 0  9990",
            "2 40378  99.1043 350.5299 0153633 201.4233 158.0516 15.09095095 39471"]



# Satellite object
sat = Satellite(sat_TLE[0], sat_TLE[1],'Firebird 4')
# Lightning-gettin' object
gld = GLD_file_tools(GLD_root, prefix='GLD')

# Column ind
lat_ind = 7;
lon_ind = 8;
mag_ind = 9;



# Input time
inTime = "2015-11-01T00:25:00"
td = datetime.timedelta(seconds = 30) # Maximum time back to check for lightning (pulse can be up to a minute! But how patient are we, really)
plottime = datetime.datetime.strptime(inTime,  "%Y-%m-%dT%H:%M:%S")

print plottime

# Get satellite location
sat.compute(plottime)
sat.coords.transform_to('geomagnetic')

# Get flashes within timeframe:
flashes, flash_times = gld.load_flashes(plottime, td)
lats = [f[lat_ind] for f in flashes]
lons = [f[lon_ind] for f in flashes]

flash_coords = coordinate_structure(lats, lons, np.zeros(np.size(lats)),'geographic')
# flash_coords.transform_to('geomagnetic')


# (Use these to make a nice grid)
lats = np.linspace(-90,90,90)
lons = np.linspace(-180,180,90)
flash_coords = coordinate_structure(lats, lons, [0],'geomagnetic')

print "%g flashes (pre-filter)" % flash_coords.len()
atten_factors = longitude_scaling(flash_coords, sat.coords)
mask = atten_factors < 24

#mask = (np.abs(flash_coords.lon() - sat.coords.lon()) < 20)
#mask = ionoAbsorp(flash_coords.lat(),4000) < 10

#print "%g flashes (post-filter)" % np.sum(mask)

mLats = flash_coords.data[mask,0]
mLons = flash_coords.data[mask,1]
masked_coords = coordinate_structure(mLats, mLons, np.zeros(np.size(mLats)),'geomagnetic')
#masked_coords.transform_to('geographic')

#flash_coords.transform_to('geographic')
#sat.coords.transform_to('geographic')

#masked_coords = coordinate_structure(flash_coords.lat()[mask], flash_coords.lon()[mask], np.zeros(np.size(mask)),'geomagnetic')

plt.figure()
plt.scatter(flash_coords.lon(),flash_coords.lat(),marker='.',color='blue')
plt.scatter(sat.coords.lon(),sat.coords.lat(),marker='o',color='green')
plt.scatter(masked_coords.lon(), masked_coords.lat(),marker='x',color='red')
plt.xlim([-180,180])
plt.ylim([-90,90])

plt.figure()
plt.plot(-1.0*ionoAbsorp(np.linspace(-90,90,90),4000))
plt.show()