import matplotlib as mpl
mpl.use('Agg')

import numpy as np
import pickle
import datetime
import os
import itertools
import random
import fluxMDP
import string



# Random run:
name = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))
gActs = np.random.choice([['off','continuous','low','mid','high'],['off','continuous']])
num_times = np.random.choice([1,4])
smoothing_radius = np.random.choice([2000,3000,4000])
switching_penalty = np.random.choice([0,0.2,0.5,1])
greed_rate = 0.5
alpha = np.random.choice([0.8,0.9,0.95])
gamma = np.random.choice([0.1, 0.25, 0.5])
storage_penalty = np.random.choice([0.5,1,2])
start_time= datetime.datetime(2015,10, 8,18,30,00)
stop_time = datetime.datetime(2015,11,17,00,00,00)






fluxMDP.fluxMDP(outDir = "outputs/random_runs/run_" + name, 
    start_time = start_time, 
    stop_time=stop_time, 
    gActs = gActs,
    num_times=num_times,
    smoothing_radius=smoothing_radius,
    switching_penalty=switching_penalty,
    greed_rate=greed_rate,
    alpha=alpha,
    gamma=gamma,
    storage_penalty=storage_penalty,
    detector_area=1000,
    previous_measurements='outputs/complete_filled_measurements.pkl')
