# Load_phi_files.py
# 11.12.2015
# Austin Sousa
#
#   Loads all Phi files from a directory
#   Returns N, S, L
#   N and S are numpy ndarrays (num_T, num_E, num_L).
#   Data are in units of [counts / (cm^2 keV s)]
#   L is a vector of L-shells found in the directory.
#
#   V1.0. Seems to match the Matlab verson!

import os
import glob
import re
#import struct
import numpy as np
#from matplotlib import pyplot as plt

def load_phi_files(dataDir, num_E=1000, num_T=600):

    allfiles = os.listdir(os.getcwd() + '/' + dataDir)

    n_files = sorted([f for f in allfiles if 'phi_' in f and '_N' in f])
    s_files = sorted([f for f in allfiles if 'phi_' in f and '_S' in f])

    if not n_files and not s_files:
        print "No phi files found!"

    #print n_files

    L_n = sorted([float(f[4:(len(f) - 2)]) for f in n_files])
    L_s = sorted([float(f[4:(len(f) - 2)]) for f in s_files])
    #L_n = sorted([float(re.findall("\d+\.\d+",f)[0]) for f in n_files])
    #L_s = sorted([float(re.findall("\d+\.\d+",f)[0]) for f in s_files])
    
    if not (L_n == L_s):
        print "North / South mismatch!"
   
    L = np.zeros(np.size(L_n))
    N = np.zeros([num_T, num_E, len(L_n)])
    #for ind, filename in enumerate(n_files):
    for ind in xrange(len(L_n)):
        filename = "phi_%g_N" % L_n[ind]
        with open(dataDir + '/' + filename, mode='rb') as file:        
            N[:,:,ind] = np.fromfile(file, np.single).reshape(num_E, num_T).T
            L[ind] = float(filename[4:(len(filename) - 2)])
            #print L[ind]

    S = np.zeros([num_T, num_E, len(L_s)])

    #for ind, filename in enumerate(s_files):
    for ind in xrange(len(L_s)):
        filename = "phi_%g_S" % L_s[ind]
        with open(dataDir + '/' + filename, mode='rb') as file:        
            S[:,:,ind] = np.fromfile(file, np.single).reshape(num_E, num_T).T


    print "Total N: %d" % len(L_n)
    print "Total S: %d" % len(L_s)
    print "N NaNs: %d" % np.sum(np.isnan(N))
    print "S NaNs: %d" % np.sum(np.isnan(S))
    return N, S, L

if __name__ == '__main__':
    # ----------------------------------
    # Main Program
    # ----------------------------------
    N, S, L = load_phi_files("outputs/run_GIOEEV/in_45", num_E = 128, num_T = 600)
