import os
import glob
import re
import numpy as np
from coordinate_structure import coordinate_structure
from matplotlib import pyplot as plt
from load_phi_files import load_phi_files
import subprocess
import re
import sys
import pickle

class flux_obj(object):
    def __init__(self):
        self.t = None
        self.coords = coordinate_structure()
        self.data = None
        self.N = None
        self.N_E = None
        self.S_E = None
        self.S = None
        self.L = None
        self.consts_list = None
        self.LK = None
        self.I0 = None
        self.RES_DT = None
        self.RES_FINT = None
        self.NUM_T = None


def build_database(input_dir_name='outputs', output_filename='database.pkl'):

    ev2joule = (1.60217657)*1e-19 # Joules / ev
    joule2millierg = 10*1e10 

    print ""
    rootDir = os.getcwd() + '/' + input_dir_name + '/'
    d = os.listdir(rootDir)

    runs = sorted([f for f in d if 'run_' in f])
    print runs

    database = dict()

    for r in runs:
        d = os.listdir(rootDir + "/" + r)
        ins = sorted([f for f in d if 'in_' in f])
        print "Run: ", r    

        consts_file = r + "/codesrc/consts.h"

        # Parse consts.h
        # Loads as many lines from the constants file as it can... will fail on
        consts_list = []
        with open(rootDir + consts_file,'r') as file:
            for line in file:
                if "#define" in line and line[0:1] is not "//":
                    l = line.split()
                    #if len(l)>=3:
                    try:
                        exec('%s=%f'% (l[1], eval(l[2])))
                        consts_list.append(l)
                        #print l[1], eval(l[1])
                    except:
                        #print "failed: ", l
                        None
        for i in ins:

            # Load some actual data!
            # try:
                inp_lat = int(i[3:])
                NUM_T = RES_FINT/RES_DT
                obj = dict() 
                #obj = flux_obj()
                
                N, S, L = load_phi_files( input_dir_name + "/" + r + "/" + i, num_E = NUM_E, num_T = NUM_T)
                
                obj['consts_list'] = consts_list
                #obj.consts_list = consts_list
                NUM_L = len(L)

                obj['NUM_T'] = NUM_T
                obj['I0'] = I0
                obj['LK'] = LK
                obj['RES_DT'] = RES_DT
                obj['RES_FINT'] = RES_FINT

                # obj.NUM_T = NUM_T
                # obj.I0 = I0
                # obj.LK = LK
                # obj.RES_DT = RES_DT
                # obj.RES_FINT = RES_FINT

                # # Scale by energy bin values: 
                # E_EXP_BOT = np.log10(E_MIN)
                # E_EXP_TOP = np.log10(E_MAX)
                # DE_EXP    = ((E_EXP_TOP - E_EXP_BOT)/NUM_E)

                # E_EXP = E_EXP_BOT + np.linspace(1,NUM_E,NUM_E)*DE_EXP
                # E = np.power(10,E_EXP)

                # E_scaled = ev2joule*joule2millierg*E
                # tmp = np.tile(E_scaled.T, [NUM_T, 1]).T
                # N_scaled = N*np.tile(tmp, [NUM_L,1,1]).T
                # S_scaled = S*np.tile(tmp, [NUM_L,1,1]).T
                
                # N_totals = np.sum(N_scaled,axis=1)
                # S_totals = np.sum(S_scaled,axis=1)
                N_totals = np.sum(N,axis=1)
                S_totals = np.sum(S,axis=1)

                # Load each (lat, Lk, I0, L-shell) vector separately
                coords = coordinate_structure(L,[-10],[0],'L_dipole')
                coords.transform_to('geomagnetic')


                # Total flux vs time and latitude
                obj['N'] = N_totals
                obj['S'] = S_totals
                # obj.N = N_totals
                # obj.S = S_totals

                # 3D array, flux vs time, latitude, energy
                obj['N_E'] = N
                obj['S_E'] = S

                obj['coords'] = coords
                obj['t'] = np.linspace(0,RES_FINT, NUM_T)
                key = (inp_lat)
                database[key] = obj


# ------------------- for single row entries
                # for ind, val in enumerate(coords.lat()):
                #     #print val
                #     obj.N = N_totals[:,ind]
                #     obj.S = S_totals[:,ind]
                #     #obj.L = val
                #     obj.Lat = round(100*val)/100.0

                #     print np.shape(obj.N)
                #     key = (inp_lat, LK, I0, round(100*val)/100)

                #     database[key] = obj

                # N_totals = np.maximum(-1000, np.log10(np.sum(N_scaled,axis=1)))
                # S_totals = np.maximum(-1000, np.log10(np.sum(S_scaled,axis=1)))


                print inp_lat
        
                # key = (inp_lat, LK, I0)
                # database[key] = obj
            # except:
            #     print "bruh ;_;"


    #print [o[1].L for o in database.items()]
    print "Saving database"
    with open(output_filename,'wb') as f:
        pickle.dump(database,f,pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    build_database(input_dir_name="outputs/probably/",output_filename="database_dicts.pkl")

