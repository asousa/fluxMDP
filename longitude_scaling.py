# Return longitude scaling for a set of inputs
import numpy as np 
import geopy.distance
from ionoAbsorp import ionoAbsorp
from coordinate_structure import coordinate_structure

def longitude_scaling(flash_coords, out_coords):
    ''' Returns dB attenuation of a wave, based on the WIPP ionosphere input model '''

    H_IONO = 1e5
    # H_E = 5000.0
    # Z0  = 377.0
    # A = 5e3
    # B = 1e5
    path_atten = 12 # db per 1000 km (Completely handwaved, but I can't get the trig from the C code right)

    R_earth = geopy.distance.EARTH_RADIUS
    D2R = np.pi/180.0

    #f = 4000 #Hz  
    # Ionospheric absorption at output points
    #iono_atten = ionoAbsorp(flash_coords.lat(),f)

    # Separation in longitude, kilometers
    dist_lon = (R_earth)*np.abs(flash_coords.lon() - out_coords.lon())*D2R*np.cos(D2R*flash_coords.lat())*1e-3
    #return path_atten*dist_lon/1000.0# - iono_atten

    return dist_lon*path_atten

    # Approx attenuation factor (db per 1000 km):
    # Note: Still ignoring vertical propagation losses





if __name__ == '__main__':
    fc1 = coordinate_structure(45,45,0,'geographic')
    fc2 = coordinate_structure(45,50,0,'geographic')    

    print longitude_scaling(fc1, fc2)