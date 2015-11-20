import numpy as np
import pickle
from build_database import flux_obj
from scipy import interpolate
# from sklearn.svm import SVR
# from sklearn.svm import NuSVR
from matplotlib import pyplot as plt
from coordinate_structure import coordinate_structure

def plot_precip(x, y, data):
    plt.figure()
    plt.pcolor(x, y, np.log10(data.T))
    plt.clim([-4, 4])


if __name__ == "__main__":

    with open('database_multiple.pkl','rb') as file:
        db = pickle.load(file)

    print db.keys()

    for lat in db.keys():
        x = db[lat].t
        y = db[lat].coords.lat()
        data = db[lat].N
        plot_precip(x,y,data)
        plt.title(lat)
        plt.show()

    # px = int(np.ceil(np.sqrt(np.size(db.keys()))))
    # py = int(np.ceil(np.sqrt(np.size(db.keys()))))

    # f2, axarr = plt.subplots(px,py)
    # axa = axarr.flatten()
    # for ind, ll in enumerate():
    #     axa[ind].plot(t,N_totals[:,ind])
    #     axa[ind].set_ylim([-4,2])
    #     axa[ind].set_xlabel(ll)

    # plt.setp([a.get_xticklabels() for a in axarr[:,:].flatten()], visible=False)
    # plt.setp([a.get_yticklabels() for a in axarr[:,:].flatten()], visible=False)

    # plt.show()
