import numpy as np
import pickle
from build_database import flux_obj
from GLD_file_tools import GLD_file_tools
from satellite import Satellite
import datetime
import ephem
from coordinate_structure import coordinate_structure
from coordinate_structure import transform_coords
from longitude_scaling import longitude_scaling
import os
from precip_model import precip_model

class measurement_model(object):
    '''Instantiate this bad boy to make precipitation measurements'''
    
    def __init__(self,
                 database='database.pkl',
                 GLD_root = 'alex/array/home/Vaisala/feed_data/GLD',
                 multiple_bands=False):
    
        self.m = precip_model(database, multiple_bands)
        
        self.RES_DT = self.m.db[self.m.db.keys()[0]].RES_DT
        self.RES_FINT = self.m.db[self.m.db.keys()[0]].RES_FINT
        
        
        self.td = datetime.timedelta(seconds = 5 + self.RES_FINT) # Maximum previous time to examine flashes in. 
                                                                  # Ideally 2*self.RES_INT to assure we don't miss anything.
        
        # Lightning database
        self.gld = GLD_file_tools(GLD_root, prefix='GLD')
        # Column indices
        self.lat_ind = 7;
        self.lon_ind = 8;
        self.mag_ind = 9;
        
    def get_measurement(self, in_time, coordinates, mode='continuous', bands=None):
        # Get flashes within timeframe:
        flashes, flash_times = self.gld.load_flashes(in_time, self.td)
        flashes = flashes[:,(self.lat_ind, self.lon_ind, self.mag_ind, self.mag_ind)]
        flash_coords = transform_coords(flashes[:,0], flashes[:,1], np.zeros_like(flashes[:,0]), 'geographic', 'geomagnetic')
        flashes[:,:2] = flash_coords[:,:2]
        flashes[:,3] = [(in_time - s).microseconds*1e-6 + (in_time - s).seconds for s in flash_times]

        
        # So! No we have an array of relevant flashes -- lat, lon, mag, time offset.
        # Let's model the flux at the satellite.
        flux = 0

        time_sampling_vector = np.linspace(-self.RES_FINT,0,np.round(self.RES_FINT/self.RES_DT))
        if mode=='continuous':
            for f in flashes:
                #print td.seconds - f[3]   
                flux += np.sum( self.m.get_precip_at(f[0], coordinates.lat(), time_sampling_vector + f[3]) *
                          self.m.get_longitude_scaling(f[0], f[1], coordinates.lon(), I0=f[2]) * self.RES_DT )
#                flux += np.sum(temp)
                #print temp
                
        if mode=='banded':
            for f in flashes:
                flux += np.sum(( np.array([self.m.get_multiband_precip_at(f[0],
                        coordinates.lat(), energy,
                        time_sampling_vector + f[3]) for energy in bands]) *
                        self.m.get_longitude_scaling(f[0], f[1], coordinates.lon(), I0=f[2]) * self.RES_DT ))
#                flux += np.sum(temp)

        
        
        return flux

if __name__== "__main__":
# -------------- Here's how to create a satellite and take some flux measurements: -------------
    GLD_root  = 'alex/array/home/Vaisala/feed_data/GLD'
    NLDN_root = 'alex/array/home/Vaisala/feed_data/NLDN'

    sat_TLE  = ["1 40378U 15003C   15293.75287141  .00010129  00000-0  48835-3 0  9990",
                "2 40378  99.1043 350.5299 0153633 201.4233 158.0516 15.09095095 39471"]

    # Satellite object:
    sat = Satellite(sat_TLE[0], sat_TLE[1],'Firebird 4')

    # Measurement object:
    f = measurement_model(database = "database_test.pkl", multiple_bands = True)

    # ---- Do The Thing:
    inTime = "2015-11-01T00:45:00"
    plottime = datetime.datetime.strptime(inTime,  "%Y-%m-%dT%H:%M:%S")

    sat.compute(plottime)
    sat.coords.transform_to('geomagnetic')

    #print "From banded measurement (all on):"
    #print f.get_measurement(plottime, sat.coords, mode='banded',bands=f.m.E_bands)
    print "From single measurement:"
    print f.get_measurement(plottime, sat.coords, mode='continuous',bands=f.m.E_bands)


