import matplotlib as mpl
mpl.use('Agg')

import numpy as np
import pickle
from build_database import flux_obj
# from scipy import interpolate
# from matplotlib import pyplot as plt
# from GLD_file_tools import GLD_file_tools
# from satellite import Satellite
import datetime
# import ephem
# from coordinate_structure import coordinate_structure
# from coordinate_structure import transform_coords
# from longitude_scaling import longitude_scaling
# from ionoAbsorp import ionoAbsorp
import os
# from mpl_toolkits.basemap import Basemap
# from precip_model import precip_model
import itertools
# from measurement_model import measurement_model
import random
# from scaling import get_time_scaling, get_map_scaling
import fluxMDP

# ---- more options run:
name = 'full'
gActs = ['off','continuous','low','mid','high']
start_time= datetime.datetime(2015,10, 8,18,30,00)
stop_time = datetime.datetime(2015,11,22,23,00,00)

fluxMDP.fluxMDP(outDir = "outputs/run_2_" + name,
    start_time = start_time, 
    stop_time=stop_time, 
    gActs = gActs)