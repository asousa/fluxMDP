
import numpy as np
import pickle
#from build_database import flux_obj
from scipy import interpolate
# from sklearn.svm import SVR
# from sklearn.svm import NuSVR
from matplotlib import pyplot as plt
from coordinate_structure import coordinate_structure
import itertools
import geopy.distance



class precip_model(object):
    def __init__(self,database="database.pkl", multiple_bands=False):

        self.R_earth = geopy.distance.EARTH_RADIUS
        self.d2r = np.pi/180.0
        self.path_atten = -12 # db per 1000 km attenuation (approximation of decay for earth-ionsphere waveguide)

        with open(database,'rb') as file:
            self.db = pickle.load(file)

        in_lats = sorted(self.db.keys())
        self.multiple_bands = multiple_bands

        # Simple dataset using total flux -- interpolates input latitude, output latitude, time.
        N = []
        S = []
        for i in in_lats:
            obj = self.db[i]
            N.append(obj['N'])
            S.append(obj['S'])
        
        N = np.array(N).swapaxes(1,2)
        S = np.array(S).swapaxes(1,2)
        self.I0 = obj['I0']
        
        self.N_interp = interpolate.RegularGridInterpolator((in_lats, obj['coords'].lat(), obj['t']), N, fill_value=0,bounds_error=False)
        self.S_interp = interpolate.RegularGridInterpolator((in_lats, obj['coords'].lat(), obj['t']), S, fill_value=0,bounds_error=False)
    
        if multiple_bands:
            # Split into multiple bands -- input latitude, output latitude, energy, time.
            N_E = []
            S_E = []
            for i in in_lats:
                obj = self.db[i]
                N_E.append(obj['N_E'])
                S_E.append(obj['S_E'])

            N_E = np.array(N_E).swapaxes(1,3)
            S_E = np.array(S_E).swapaxes(1,3)
#            print np.shape(N_E)
            self.E_bands = np.linspace(1,np.shape(N_E)[2], np.shape(N_E)[2])
#            print self.E_bands
            self.N_E_interp = interpolate.RegularGridInterpolator((in_lats, obj['coords'].lat(), self.E_bands, obj['t']), N_E, fill_value=0,bounds_error=False)
            self.S_E_interp = interpolate.RegularGridInterpolator((in_lats, obj['coords'].lat(), self.E_bands, obj['t']), S_E, fill_value=0,bounds_error=False)


    def get_precip_at(self, in_lat, out_lat, t):
        ''' in_lat:  Flash latitude (degrees)
            out_lat: Satellite latitude (degrees)
            t:       Time elapsed from flash (seconds)
            '''
        # print in_lat
        # print out_lat
        # print t
        #inps = np.array([t for t in itertools.product(*[np.array(in_lat),np.array(out_lat),np.array(t)])])

        #inps = self.tile_keys(np.array([in_lat, out_lat]), t)
        #print "inps: ", inps

        # Model is symmetric around northern / southern hemispheres (mag. dipole coordinates):
        # If in = N, out = N  --> Northern hemisphere
        #    in = N, out = S  --> Southern hemisphere
        #    in = S, out = N  --> Southern hemisphere
        #    in = S, out = S  --> Northern hemisphere
        use_southern_hemi = (in_lat > 0) ^ (out_lat > 0)

        inps = self.tile_keys(np.array([np.abs(in_lat), np.abs(out_lat)]), t)

        if use_southern_hemi:
            return self.S_interp(inps)
        else:
            return self.N_interp(inps)


    def get_multiband_precip_at(self, in_lat, out_lat, energy, t):
        if not self.multiple_bands:
            print "No multi-band!"
        else:

            use_southern_hemi = (in_lat > 0) ^ (out_lat > 0)
            inps = self.tile_keys(np.array([np.abs(in_lat), np.abs(out_lat), energy]), t)

            # inps = self.tile_keys(np.array([in_lat, out_lat, energy]), t)
            if use_southern_hemi:
                return self.S_E_interp(inps)
            else:
                return self.N_E_interp(inps)


    def tile_keys(self, key1, key2):
        return np.vstack([np.outer(key1, np.ones(np.size(key2))), key2]).T

    def get_longitude_scaling(self, inp_lat, inp_lon, out_lon, I0=None, db_scaling = False):
        ''' inp_lat: Scalar latitude
            inp_lon: Scalar longitude 
            out_lon: vector of longitudes to compute at
        '''

        # ----------- Old version: In all your computed measurements, rats (12.2.2015) ----
        #dist_lon = (self.R_earth)*np.abs(inp_lon - out_lon)*self.d2r*np.cos(self.d2r*inp_lat)*1e-3
        #vals = dist_lon*self.path_atten

        # ----------- New version: Actually works, does wraparound properly -------
        b = np.cos(self.d2r*inp_lat)*np.sin(self.d2r*(inp_lon - out_lon)/2.0)
        dist_lon = self.R_earth*2*np.arcsin(np.abs(b))

        vals = dist_lon*self.path_atten/1000.0

        if db_scaling:
                return vals
        else:
            if not I0:
                return np.power(10,vals/10.0)
            else:
                return np.power(10,vals/10.0)*(I0**2)/(self.I0**2)
                

if __name__ =="__main__":

    m = precip_model("database_dicts.pkl",multiple_bands=True)

    t = np.linspace(0,30,300)
    tmp = m.get_multiband_precip_at(30,45,5,t)

    plt.plot(t,tmp)
    plt.show()
    #tmp = [m.get_precip_at(30,45,x,t,"N") for x in m.E_bands]
    #print tmp

    # t = np.linspace(0,30,100)
    # out_lats = np.linspace(30,70,60)
    # in_lat = 19
    # res = []

    # for o in out_lats:
    #     points = m.tile_keys((in_lat, o), t)
    #     res.append(m.N_interp(points))

    # res = np.log10(np.array(res))

    # plt.pcolor(t, out_lats, res)
    # plt.clim([-4,4])
    # plt.show()











