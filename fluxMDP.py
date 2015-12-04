# The main event! Determine optimal sampling modes for a satellite at a given location, using reinforcement learning.
# import matplotlib as mpl
# mpl.use('Agg')


import numpy as np
import pickle
#from build_database import flux_obj
from scipy import interpolate
import matplotlib.pyplot as plt
#from matplotlib import pyplot as plt
from GLD_file_tools import GLD_file_tools
from satellite import Satellite
import datetime
import ephem
from coordinate_structure import coordinate_structure
from coordinate_structure import transform_coords
from longitude_scaling import longitude_scaling
from ionoAbsorp import ionoAbsorp
import os
#from mpl_toolkits.basemap import Basemap
from precip_model import precip_model
import itertools
from measurement_model import measurement_model
import random
#from scaling import get_time_scaling, get_map_scaling


def get_map_scaling(grid_lats, grid_lons, in_coords, Rmax=1000):
    ''' Get scaling coefficients, based on squared distance from input coordinates.
        Output is a 2d array with dimensions grid_lats, grid_lons
    '''
    d2r = np.pi/180.0
    iLat = in_coords.lat()
    iLon = in_coords.lon()
    # Great circle distance, in radians -- haversine formula. V E C T O R I Z E D
    a = np.outer(np.sin(d2r*(grid_lats - iLat)/2.0)**2, np.ones_like(grid_lons))
    b = np.outer(np.cos(d2r*iLat)*np.cos(d2r*grid_lats), np.sin(d2r*(grid_lons - iLon)/2.0)**2)
    dists = 6371*2*np.arcsin(np.sqrt(a + b))

    # Select entries around a small patch, and scale quadratically:
    weights = (np.maximum(0, Rmax - dists))**2
    # Normalize selection to 1
    return weights/np.sum(weights)
#    return weights/np.max(weights)


def get_time_scaling(grid_times, cur_coords, cur_time):
    '''Coarsely bin into a local time of day, with the idea that
    lightning has some hourly time dependence (i.e., rarely lightning in the morning)
    '''
    d2r = np.pi/180.0
    if not cur_coords.type=='geographic':
        print "Not in Geographic coordinates!"
    #    time_bins = np.linspace(0,23,4)             # Vector of times to quantize to
    d = cur_coords.lon()[0]*24/360                   # Hour shift in longitude    
    
    # Local time, in fractional hours
    LT = cur_time.hour + np.sign(d)*cur_time.minute/60.0 + d
    
    # Did we loop around a day?  (er, past the rounding point?)
    if LT < grid_times[0] -np.size(grid_times):
        LT += 24
    if LT >= grid_times[-1] + np.size(grid_times):
        LT -= 24

    dists = np.abs(LT - grid_times)
    weights = (24 - dists)**2
    return weights/np.sum(weights)

def fluxMDP(start_time = datetime.datetime(2015,11,01,01,45,00),
            stop_time = datetime.datetime(2015,11,1,2,45,00),
            storage_penalty = 1,
            detector_area = 1e3,
            switching_penalty = 0.05,
            smoothing_radius = 1000,
            alpha = 0.9,
            gamma = 0.1,
            greed_rate = 2,
            fixed_greed = None,
            outDir = 'MDP_saves',
            gActs  = ['continuous','off'],
            previous_measurements = None,
            stored_policy = None):

    db_name = "database_dicts.pkl"

    print "Database name: ", db_name
    print "start time: ", start_time
    print "stop time: ", stop_time
    print "storage penalty: ", storage_penalty
    print "switching penalty: ", switching_penalty
    print "smoothing radius: ", smoothing_radius
    print "alpha: ", alpha
    print "gamma: ", gamma
    print "fixed_greed: ",fixed_greed
    print "greed rate: ", greed_rate
    print "outDir: ",outDir
    print "Actions: ", gActs



    if previous_measurements:
        print "Loading previous measurements file " + previous_measurements
        using_previous = True
        with open(previous_measurements,'rb') as file:
            prev_db = pickle.load(file)
    else:
        using_previous = False


    # ------------------- Initial setup --------------------
    # GLD_root  = 'alex/array/home/Vaisala/feed_data/GLD'
    # NLDN_root = 'alex/array/home/Vaisala/feed_data/NLDN'
    GLD_root = 'GLD'

    sat_TLE  = ["1 40378U 15003C   15293.75287141  .00010129  00000-0  48835-3 0  9990",
                "2 40378  99.1043 350.5299 0153633 201.4233 158.0516 15.09095095 39471"]

    # Satellite object:
    sat = Satellite(sat_TLE[0], sat_TLE[1],'Firebird 4')

    # Measurement object:
    f = measurement_model(database = db_name, GLD_root=GLD_root, multiple_bands = True)

    # Start time:
    # start_time = "2015-11-01T00:45:00"
    tStep = datetime.timedelta(seconds=30) # Step size thru model
    cur_time = start_time
    mid_time = start_time + datetime.timedelta(days=15) # Midpoint of greediness-increasing curve
    max_greed = 0.95  # Maximum greed asypmtote
    #cur_time = datetime.datetime.strptime(start_time,  "%Y-%m-%dT%H:%M:%S")

    # Stop time:
    #stop_time = datetime.datetime(2015,11,1,2,45,00)



    # # State space:
    # gLats  = np.linspace(-90,90,90)
    # gLons  = np.linspace(-180,180,180)
    # #gTimes = np.linspace(0,24,4)
    # gTimes = np.linspace(0,24,1)
    # # gActs  = ['continuous','off','low','mid','high']

    bands = dict()
    bands['low'] = [1,2]
    bands['mid'] = [3, 4, 5]
    bands['high']= [6, 7, 8]


    # # Tweaking parameters:
    # storage_penalty = 1
    # switching_penalty = 0.05
    # alpha = 0.9
    # gamma = 0.1
    # greed_rate = 2 # Percent increase per four hours

    # greed = 0.01
    if fixed_greed:
        greed = fixed_greed
    else:
        greed = 0


    # outDir = 'MDP_saves'
    if not os.path.exists(outDir):
        os.makedirs(outDir)

    reward_table = []

    # Q matrix
    if stored_policy:
        print "Using previously-computed policy"
        using_stored_policy = True
        with open(stored_policy,'rb') as file:
            Q = pickle.load(file)

    else:  
        using_stored_policy = False
        Q = np.zeros([90, 180, 1, np.size(gActs)])

    # State space:
    gLats  = np.linspace(-90,90,np.shape(Q)[0])
    gLons  = np.linspace(-180,180,np.shape(Q)[1])
    gTimes = np.linspace(0,24,np.shape(Q)[2])



    # Start a file to periodically dump entries to:
    odb = dict()
    odb['lats']    = gLats
    odb['lons']    = gLons
    odb['times']   = gTimes
    odb['actions'] = gActs
    odb['bands']   = bands

    with open(outDir + '/odb.pkl','wb') as file:
        pickle.dump(odb, file)


    sat.compute(cur_time)

    # Get time scaling weights:
    time_weight = get_time_scaling(gTimes, sat.coords, cur_time)
    sat.coords.transform_to('geomagnetic')
    # Get distance interpolating weights:
    map_weights = get_map_scaling(gLats, gLons, sat.coords, Rmax=smoothing_radius)
    # 3d weight (lat, lon, time):
    W = map_weights[:,:,None]*time_weight
    print "W isnans: ",np.sum(np.isnan(W))
    print "Q isnans: ",np.sum(np.isnan(Q))

    action = gActs[0]

    print "Starting run from ", cur_time
    ind = 0 # iteration counter
    #for ind in range(100):
    while cur_time < stop_time:
        #try:
            #print "i: ", ind
            #print "mod i: ", np.mod(ind,10)
            # select an action
            prev_action = action

            brains = np.random.choice(['greedy','adventurous'],p=[greed, 1.0-greed])
            print "Greed factor: ", greed 
            # ------------------- Use previously-computed measurements only: ---------------------
            if using_previous:
                if cur_time in prev_db:
                    avail_measurements = prev_db[cur_time].keys()
                    Q_inds = [gActs.index(a) for a in avail_measurements if a in gActs]
                    print "available measurements: ", avail_measurements

                    if brains == 'greedy':
                        a_tmp = np.argmax([np.sum(Q[:,:,:,i]*W) for i in Q_inds])
                        a = Q_inds[a_tmp]
                        action = gActs[a]
                    elif brains == 'adventurous':
                        a = np.random.choice(Q_inds)
                        action = gActs[a]
                
                    meas = prev_db[cur_time][action]

                    if action =='off':
                        reward = 0 - switching_penalty*(not(action==prev_action))
                    elif action =='continuous':
                        reward = meas*detector_area - storage_penalty - switching_penalty*(not(action==prev_action))
                    elif action in ['low','mid','high']:
                        reward = meas*detector_area - (len(bands[action])/8.0)*storage_penalty- switching_penalty*(not(action==prev_action))
                else:
                    # Missing stored data --- escape current iteration, increment clock, try again:
                    cur_time += tStep
                    continue

            else: 
            # ------------------ Compute fresh measurements: -------------------------------------

                if brains =='greedy':
                    a = np.argmax([np.sum(Q[:,:,:,i]*W) for i in range(len(gActs))])
                    action = gActs[a]
                elif brains =='adventurous':
                    action = np.random.choice(gActs)
                    a = gActs.index(action)

                #print "Feeling", brains,":",action

                #action = 'continuous' #random.choice(gActs)
                #a = gActs.index(action)
                #print action
                # take a measurement, calculate reward:
                if action =='off':
                    meas = 0
                    reward = 0 - switching_penalty*(not(action==prev_action))
                
                if action =='continuous':
                    meas = np.sqrt(f.get_measurement(cur_time, sat.coords, mode='continuous'))
                    # reward = meas*detector_area - storage_penalty*(action not in ['off']) - switching_penalty*(not(action==prev_action))
                    reward = meas*detector_area - storage_penalty - switching_penalty*(not(action==prev_action))

                if action in ['low','mid','high']:
                    meas = np.sqrt(f.get_measurement(cur_time, sat.coords, mode='banded',bands=bands[action]))
                    # reward = meas*detector_area - (len(bands[action])/8.0)*storage_penalty*(action not in ['off']) - switching_penalty*(not(action==prev_action))
                    reward = meas*detector_area - (len(bands[action])/8.0) - switching_penalty*(not(action==prev_action))




            #cur_state_continuous = [sat.coords.lat()[0], sat.coords.lon()[0], cur_time,  action]
            cur_state_continuous = [sat.coords, cur_time,  action]
            
            # Get Q(t,a)
            Qcur = np.sum(Q[:,:,:,a])*W

            print "Qcur isnans: ",np.sum(np.isnan(Qcur))

            # increment timestep:
            cur_time += tStep
            
            # Update satellite position for t+1:
            sat.compute(cur_time)
            #geo_coords = sat.coords  # Save geographic longitude for time binning on the next iteration

            # Get time scaling weights at t+1:
            time_weight_next = get_time_scaling(gTimes, sat.coords, cur_time)
            # Back to geomagnetic (time weights need geographic):
            sat.coords.transform_to('geomagnetic')
            # Get distance interpolating weights at t+1:
            map_weights_next = get_map_scaling(gLats, gLons, sat.coords, Rmax=smoothing_radius)
            # 3d weight (lat, lon, time):
            W_next = map_weights_next[:,:,None]*time_weight_next


            # Get max{a} Q(t+1,a):
        #    Qmax = np.max([np.sum(Q[:,:,i]*map_weights) for i in range(len(gActs))])
            Qmax = np.max([np.sum(Q[:,:,:,i]*W_next) for i in range(len(gActs))])

            print "Qmax: ",Qmax
            
            # update Q    
            # tmp2 = alpha*(reward + gamma*Qmax - Qcur)*map_weights
            # print "tmp2 is: ",np.shape(tmp2)
            #Q[:,:,a] = Q[:,:,a] + alpha*(reward + gamma*Qmax - Qcur)*map_weights
            Q[:,:,:,a] = Q[:,:,:,a] + alpha*(reward + gamma*Qmax - Qcur)*W
            
            # Rename the weights for the next round:
            W = W_next
            # map_weights = map_weights_next
            # time_weight = time_weight_next

            # Store the current state, action, and reward
            cv = [cur_state_continuous, meas, reward]
            print cv
            reward_table.append(cv)

            # Increment greediness:
            if not fixed_greed:
                greed = max_greed/(1 + np.exp(-greed_rate*((cur_time - mid_time).total_seconds()/(24*3600))))

            # if (np.mod(cur_time.hour, 4)==0) and (cur_time.minute == 0) and (cur_time.second == 0):
            #     # Get greedier:
            #     print "Getting greedier..."
            #     greed = greed*(1 + greed_rate/100.0) 
            

            if (np.mod(cur_time.day,2)==0) and (cur_time.hour ==0) and (cur_time.minute == 0) and (cur_time.second == 0):
                print "Saving progress..."
                # Archive where we're at:
                with open(outDir + '/data_i%g.pkl' % ind,'wb') as file:
                    pickle.dump(reward_table,file)
                    reward_table = []

                with open(outDir + '/Q_i%g.pkl' % ind,'wb') as file:
                    pickle.dump(Q,file)
                    

                ax_x = int(np.ceil(np.sqrt(len(gTimes))))
                ax_y = 1

                while (ax_x*ax_y < len(gTimes)):
                    ax_y += 1

                # Q plots

                Q_clims = [np.min(Q), np.max(Q)]
                for act in gActs:
                    fig, ax = plt.subplots(ax_x, ax_y)

                    if len(gTimes) == 1:
                        ax.pcolor(gLons, gLats, Q[:,:,0,gActs.index(act)]/np.max(Q),clim=Q_clims)
                        ax.set_title(gTimes[0])
                        ax.axis('off')
                        ax.scatter(sat.coords.lon(),sat.coords.lat(),marker='x')
                        fig.suptitle('action: %s, %g iterations: \n%s' % (act, ind, cur_time))
                    else:

                        ax = ax.flat
                        for x in range(len(gTimes)):
                            ax[x].pcolor(gLons, gLats, Q[:,:,x,gActs.index(act)]/np.max(Q), clim=Q_clims)
                            ax[x].set_title(gTimes[x])
                            ax[x].axis('off')
                            ax[x].scatter(sat.coords.lon(),sat.coords.lat(),marker='x')
                            fig.suptitle('action: %s, %g iterations: \n%s' % (act, ind, cur_time))
                             
                    figname = outDir + '/Q_%g_%s.png' % (ind, act)
                    print "Save filename: ", figname
                    plt.savefig(figname)

                plt.close(fig)

                # Policy plot
                policy = np.argmax(Q, axis=3)
                #print np.shape(policy)
                #print np.min(policy)

                fig, ax = plt.subplots(ax_x, ax_y)

                if len(gTimes) == 1:
                    ax.pcolor(gLons, gLats, policy[:,:,0])
                    ax.set_title(gTimes[0])
                    ax.axis('off')

                else:
                    ax = ax.flat

                    for x in range(np.size(gTimes)):
                        ax[x].pcolor(gLons, gLats, policy[:,:,x])
                        ax[x].set_title(gTimes[x])
                        ax[x].axis('off')
                        fig.suptitle('Policy Evaluation: %g iterations: \n%s' % (ind, cur_time))

                figname = outDir + '/policy_%g_%s.png' % (ind, act)
                print "save filename: ", figname
                plt.savefig(figname)
                plt.close(fig)

        # except:
        #     print "Something messed up! Trying the next step"
        #     cur_time += tStep

            ind += 1

if __name__ =='__main__':
    fluxMDP(gActs=['off','continuous'])
