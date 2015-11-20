# Plot an event from the saved SVR:

import numpy as np
import pickle
from build_database import flux_obj
from sklearn.svm import SVR
from sklearn.svm import NuSVR
from matplotlib import pyplot as plt
from coordinate_structure import coordinate_structure


with open('S.pkl','rb') as file:
    S = pickle.load(file)

with open('database_lat.pkl','rb') as file:
    db = pickle.load(file)


L = []
for k in db.keys():
    #print k[3]
    L.extend([k[3]])

L = sorted(np.unique(L))
print L
lat = 30
LP = 5.39
I0 = -200000.0

t = np.linspace(0,30,600)

N = []
for ind, val in enumerate(L):

    a = np.outer([lat,LP,I0, val], np.ones(600))
    b = t
    # print np.shape(a)
    # print np.shape(b)
    X = np.vstack([ a, b ]).T

    tmp = S.predict(X)
    print np.shape(tmp)
    N.append(tmp)

print np.shape(N)
N = np.maximum(0.0, N)

N_log = np.maximum(-100, np.log10(N))


cs = coordinate_structure(L,[-10],[0],'L_dipole')
cs.transform_to('geomagnetic')
print cs.lat()
plt.figure()
plt.pcolor(t,np.flipud(cs.lat()), N_log)
plt.colorbar()
plt.show()
