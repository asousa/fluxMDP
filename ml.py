# Let's try some machine learning! On a very small set!

import numpy as np
import pickle
from build_database import flux_obj
from sklearn.svm import SVR
from sklearn.svm import NuSVR
from matplotlib import pyplot as plt

with open('database_lat.pkl','rb') as file:
    db = pickle.load(file)


print db.keys()


S = NuSVR(kernel='rbf')

X = []
Y = []
for k in db.keys():
#k = db.keys()[5]
#    print np.array(k)
    t = np.linspace(0,db[k].RES_FINT,db[k].NUM_T)
    #X = np.atleast_2d(t).T
    #Y = np.power(10,db[k].N)
    inp = np.vstack([np.outer(np.array([k[0],k[3]]), np.ones(int(db[k].NUM_T))), t]).T
    X.extend(inp)
    Y.extend(np.power(10,db[k].N))
    #Y.extend(db[k].N)


ramp = np.arange(np.size(Y))
np.random.shuffle(ramp)

n_train = 50000
#n_train = np.size(Y) - 1000
n_test = 10000
train_mask = ramp[0:n_train]
test_mask = ramp[n_train+1:n_train + n_test + 1]

# print "Max ramp: ", np.max(ramp)
# print "Max shuf: ", np.max(ramp)


# plt.plot(train_mask)
# plt.show()
# print np.shape(train_mask)

#    print np.shape(inp)
    #inp2 = np.vstack([inp, t])
X = np.array(X)
Y = np.array(Y)
print "X: ", np.shape(X)
print "Y: ", np.shape(Y)
print "isnans: ", np.sum(np.isnan(Y))

print "Fitting to S..."
S.fit(X[train_mask,:],Y[train_mask])

print "Saving S "

with open('S.pkl','wb') as file:
    pickle.dump(S, file, pickle.HIGHEST_PROTOCOL)

# plt.figure()
plt.plot(Y[test_mask],color='red',marker='.')
plt.plot(S.predict(X[test_mask,:]))
plt.show()



