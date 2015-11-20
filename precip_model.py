
import numpy as np
import pickle
from build_database import flux_obj
from scipy import interpolate
# from sklearn.svm import SVR
# from sklearn.svm import NuSVR
from matplotlib import pyplot as plt
from coordinate_structure import coordinate_structure
import itertools
import geopy.distance



class precip_model(object):
    def __init__(self,database="database.pkl", energies=False):
        self.R_earth = geopy.distance.EARTH_RADIUS
        self.d2r = np.pi/180.0
        self.path_atten = -12 # db per 1000 km attenuation (approximation of decay for earth-ionsphere waveguide)

        with open(database,'rb') as file:
            self.db = pickle.load(file)

        in_lats = sorted(self.db.keys())

        N = []
        S = []
        if not energies:
            # Simple dataset using total flux -- interpolates input latitude, output latitude, time.

            for i in in_lats:
                obj = self.db[i]e
                N.append(obj.N)
                S.append(obj.S)

            N = np.array(N).swapaxes(1,2)
            S = np.array(S).swapaxes(1,2)
            self.I0 = obj.I0
            
            self.N_interp = interpolate.RegularGridInterpolator((in_lats, obj.coords.lat(), obj.t), N, fill_value=0,bounds_error=False)
            self.S_interp = interpolate.RegularGridInterpolator((in_lats, obj.coords.lat(), obj.t), S, fill_value=0,bounds_error=False)
        
        else:
            # More-complicated: Interpolates input latitude, output latitude, time, energy.
            

    def get_precip_at(self, in_lat, out_lat, t, hemisphere = "N"):
        ''' in_lat:  Flash latitude (degrees)
            out_lat: Satellite latitude (degrees)
            t:       Time elapsed from flash (seconds)
            '''
        # print in_lat
        # print out_lat
        # print t
        #inps = np.array([t for t in itertools.product(*[np.array(in_lat),np.array(out_lat),np.array(t)])])

        inps = self.tile_keys(np.array([in_lat, out_lat]), t)

        #print "inps: ", inps
        if hemisphere=="N":
            return self.N_interp(inps)
        else:
            return self.S_interp(inps)

    def tile_keys(self,key1, key2):

        return np.vstack([np.outer(key1, np.ones(np.size(key2))), key2]).T

    def get_longitude_scaling(self, inp_lat, inp_lon, out_lon, I0=None, db_scaling = False):
        ''' inp_lat: Scalar latitude
            inp_lon: Scalar longitude 
            out_lon: vector of longitudes to compute at
        '''

        dist_lon = (self.R_earth)*np.abs(inp_lon - out_lon)*self.d2r*np.cos(self.d2r*inp_lat)*1e-3
        vals = dist_lon*self.path_atten

        if db_scaling:
                return vals
        else:
            if not I0:
                return np.power(10,vals/10.0)
            else:
                return np.power(10,vals/10.0)*(I0**2)/(self.I0**2)
                

if __name__ =="__main__":

    m = precip_model("database_multiple.pkl")

    t = np.linspace(0,30,100)
    out_lats = np.linspace(30,70,60)
    in_lat = 19
    res = []

    for o in out_lats:
        points = m.tile_keys((in_lat, o), t)
        res.append(m.N_interp(points))

    res = np.log10(np.array(res))

    plt.pcolor(t, out_lats, res)
    plt.clim([-4,4])
    plt.show()










