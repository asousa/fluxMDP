# coordinate_structure.py
# Austin Sousa / 11.2015
#
# A basic class for dealing with geographic coordinates. Data is stored in a Numpy ndarry, with entries corresponding to rows.
# Basic usage:
#   cs = coordinate_struct()
#   cs.set_coords(lats, lons, alts, 'geographic')
#   cs.set_coords(L, lons, alts, 'L_dipole')
#   cs.set_cooords(x,y,z,'ecef')
#   cs.transform_to('L_dipole')
# 
# ...etc. If the three inputs are of different length, all unique combinations will be stored.
#           
# 
# To do: Make sure I did all the transformations right.
#        Use Cython / numexpr / etc for speed boost
#        Incorporate pyproj / pysatel / some other coordinate structure format

import numpy as np
import itertools
#import geopy.distance
#import numexpr as ne


class coordinate_structure(object):
    ''' Coordinate-holding object'''
    def __init__(self, in1= None, in2 = None, in3 = None, datatype=None):
        self.type = None
        self.data = None
        self.xtra_data = None
        # Radius of the earth
        self.R_earth = 6.3710e6 # Meters
        #self.R_earth = geopy.distance.EARTH_RADIUS*1e3

        # Radians, degrees
        self.d2r = np.pi/180.0
        self.r2d = 180.0/np.pi

        # Degree rotations between geographic and geomagnetic
        self.theta = -10.46;         # Shift in theta
        self.phi   = 71.57;          # Shift in phi
    

        self.valid_types = ["geographic", "geomagnetic", "L_dipole", "ecef"]
        
        # If we have data, initialize
        if (not in1 is None) and (not in2 is None) and (not in3 is None) and (datatype):
#        if (not in1 is None) and (not in2 is None) and (not in3 is None) and (datatype):
            # print "Have data!"
            # print datatype
            self.set_coords(in1, in2, in3, datatype)


    def set_coords(self, in1, in2, in3, datatype):

        if not datatype in self.valid_types:
            print "invalid data type!"
            return
        self.type = datatype

        unique_counts = np.unique([np.size(in1),np.size(in2),np.size(in3)])

        if len(unique_counts) == 1:
            # easy
            #print "single"
            if np.size(in1)==1 and np.size(in2) == 1 and np.size(in3) == 1:
                self.data = np.atleast_2d([in1, in2, in3])
            else:
                self.data = np.array([np.array(in1), np.array(in2), np.array(in3)]).T
        else:
            #print "multing"
            # Enumerate all possible combinations of the input data:
            # (This works great, but it's slow! An explicit solution might work better)
            self.data = np.array([t for t in itertools.product(*[np.array(in1),np.array(in2),np.array(in3)])])
            self.data.view('i8,i8,i8').sort(order=['f1'],axis=0)
            
        #print self.data
        #print "Loaded data shape: ", np.shape(self.data)


    def transform_to(self,target_type):

        if not target_type in self.valid_types:
            print "invalid data type!"
            return

        if target_type == self.type:
            print "No rotation needed"
            return

        # First, rotate to ECEF:
        #print self.data
        self.rotate_to_ECEF()
        #print self.data
        self.rotate_from_ECEF(target_type)
        #print self.data
        
    def rotate_to_ECEF(self):
        '''Rotate from current coords to ECEF'''
        if self.type == "geographic":
            lats = self.data[:,0]
            lons = self.data[:,1]
            alts = self.data[:,2]

            self.data = self.spherical_to_cartesian(lats, lons, alts)

        if self.type == "geomagnetic":
            lats = self.data[:,0]
            lons = self.data[:,1]
            alts = self.data[:,2]

            xyz = self.spherical_to_cartesian(lats, lons, alts)

            # rotate to ECEF:
            R = self.rotz(-self.phi).dot(self.roty(-self.theta)).T
            self.data = xyz.dot(R)
            # self.data = self.rotz(-self.phi).dot(self.roty(-self.theta)).dot(xyz)

        if self.type == "L_dipole":

            RL = np.abs(self.data[:,0])*self.R_earth
            Ro = self.R_earth + self.data[:,2]
            #print Ro
            dip_angle = np.arccos(np.sqrt(Ro/RL))   # Dip angle in radians
            hem = np.sign(self.data[:,0])           # Sign of L (hemisphere notation)

            lats = self.r2d*hem*dip_angle
            lons = self.data[:,1]
            alts = self.data[:,2]

            #print "data size: ", np.shape(self.data)
            xyz = self.spherical_to_cartesian(lats, lons, alts)
            #print "xyz size: ", np.shape(xyz)

            # rotate to ECEF:
            R = self.rotz(-self.phi).dot(self.roty(-self.theta)).T
            self.data = xyz.dot(R)
            # self.data = self.rotz(-self.phi).dot(self.roty(-self.theta)).dot(xyz)


        #print self.data
        self.type = "ecef"



    def rotate_from_ECEF(self,target_type):
        '''Rotate from ECEF to target type'''
        if not (self.type == "ecef"):
            print "not in ECEF!"
            return

        if target_type == "geographic":
            self.data = self.cartesian_to_spherical(self.data[:,0],self.data[:,1],self.data[:,2])
            self.type = target_type

        if target_type == "geomagnetic":
            R = self.roty(self.theta).dot(self.rotz(self.phi)).T
            self.data = self.data.dot(R)
            self.data = self.cartesian_to_spherical(self.data[:,0],self.data[:,1],self.data[:,2])
            self.type = target_type

        if target_type == "L_dipole":
            R = self.roty(self.theta).dot(self.rotz(self.phi)).T
            self.data = self.data.dot(R)
            self.data = self.cartesian_to_spherical(self.data[:,0],self.data[:,1],self.data[:,2])

            lats = self.data[:,0]
            lons = self.data[:,1]
            alts = self.data[:,2]

            L = np.sign(lats)*(self.R_earth + alts)/(self.R_earth*np.power(np.cos(self.d2r*lats),2))
            self.data[:,0] = L

            self.type=target_type



    def spherical_to_cartesian(self, lats, lons, alts):
        #print lats, lons, alts
        x = (alts + self.R_earth)*np.cos(self.d2r*lats)*np.cos(self.d2r*lons)
        y = (alts + self.R_earth)*np.cos(self.d2r*lats)*np.sin(self.d2r*lons)
        z = (alts + self.R_earth)*np.sin(self.d2r*lats)
        #print x, y, z
        return np.atleast_2d(np.array([x, y, z])).T

    def cartesian_to_spherical(self, x, y, z):
        alts = np.linalg.norm([x,y,z],axis=0)
        lons = self.r2d*np.arctan2(y, x);
        lats = self.r2d*np.arcsin(z/alts)
        alts = alts - self.R_earth

        return np.atleast_2d(np.array([lats, lons, alts])).T


    # Rotation matrices
    def rotz(self,theta):
        return np.array([[np.cos(theta*self.d2r), -np.sin(theta*self.d2r), 0],
                        [np.sin(theta*self.d2r),  np.cos(theta*self.d2r),  0],
                        [0                     ,  0                     , 1]]);

    # Rotation matrices
    def roty(self,theta):
        return np.array([[np.cos(theta*self.d2r),  0, np.sin(theta*self.d2r)],
                        [0                     ,  1,                       0],
                        [-np.sin(theta*self.d2r), 0,  np.cos(theta*self.d2r)]]);



    # Getters:
    def lat(self):
        if self.type in ["geographic","geomagnetic"]:
            return self.data[:,0]
        else:
            print "No latitude in current system"

    def lon(self):
        if self.type in ["geographic","geomagnetic","L_dipole"]:
            return self.data[:,1]
        else:
            print "No longitude in current system"

    def alt(self):
        if self.type in ["geographic","geomagnetic","L_dipole"]:
            return self.data[:,2]
        else:
            print "no altitude in current system"

    def len(self):
        return np.shape(self.data)[0]



    # def __iter__(self):
    #     return self

    # def next(self):
    #     if 

def transform_coords(a, b, c, source, destination):
    m = coordinate_structure(a,b,c,source)
    m.transform_to(destination)
    return m.data


if __name__ == "__main__":

    cs = coordinate_structure()

    cs.set_coords(3,10,100,"L_dipole")
    cs.transform_to("ecef")
    print cs.type
    print cs.data
    # cs.transform_to("L_dipole")
    # cs.transform_to("geographic")

