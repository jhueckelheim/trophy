'''
import pyart
import sys
import os
import time

sys.path.append('/Users/clancy/repos/trophy/python/')

#import pydda

#sys.path.append('/Users/clancy/local_packages/jaxpydda')
#import jaxpydda as pydda

#sys.path.append('/Users/clancy/local_packages/dynTRpydda')
sys.path.append('/Users/clancy/repos/trophy/python/dynTRpydda')
import dynTRpydda as pydda

#sys.path.append('/Users/clancy/local_packages/jdynTRpydda')
#import jdynTRpydda as pydda


from matplotlib import pyplot as plt

import numpy
import warnings
warnings.filterwarnings("ignore")


#numpy.show_config()

berr_grid = pyart.io.read_grid(pydda.tests.EXAMPLE_RADAR0)
cpol_grid = pyart.io.read_grid(pydda.tests.EXAMPLE_RADAR1)

sounding = pyart.io.read_arm_sonde(
    pydda.tests.SOUNDING_PATH)


# Load sounding data and insert as an intialization
u_init, v_init, w_init = pydda.initialization.make_wind_field_from_profile(
    cpol_grid, sounding[1], vel_field='corrected_velocity')




# ENTER DESIRED FILE PATH IN QUESTION

prec_vec = [1]
tol = 1e-9
subprob_tol = 1e-7
memory_size = 10
third_folder = '/home/clancy/trophy_data/SINGLE/'
detail_str = 'memory'+str(memory_size) + '_subprobtol' + "{:.0e}".format(subprob_tol)
fourth_folder = third_folder + detail_str
if not os.path.isdir(fourth_folder):
    os.system('mkdir ' + fourth_folder)




# Start the wind retrieval. This example only uses the mass continuity
# and data weighting constraints.
ti = time.time()


Grids = pydda.retrieval.get_dd_wind_field([berr_grid, cpol_grid], u_init, v_init, w_init, Co=1.0, Cm=1500.0,
                                          Cz=0, frz=5000.0, filt_iterations=0, mask_outside_opt=True, upper_bc=1,
                                          store_history=True, max_iterations=60000,
                                          max_memory=memory_size, use_dynTR=True, gtol=tol, precision_vector=prec_vec,
                                          subproblem_tol=subprob_tol, write_folder=fourth_folder)



tf = time.time()
print('Time elapsed is', tf-ti)


#### PLOTTING ROUTINE
# uncomment following to show plots

"""
# Plot a horizontal cross section
plt.figure(figsize=(9, 9))
pydda.vis.plot_horiz_xsection_barbs(Grids, background_field='reflectivity',
                                    level=6,
                                    w_vel_contours=[3, 6, 9, 12, 15],
                                    barb_spacing_x_km=5.0,
                                    barb_spacing_y_km=15.0)
plt.show()

# Plot a vertical X-Z cross section
plt.figure(figsize=(9, 9))
pydda.vis.plot_xz_xsection_barbs(Grids, background_field='reflectivity',
                                 level=40,
                                 w_vel_contours=[3, 6, 9, 12, 15],
                                 barb_spacing_x_km=10.0,
                                 barb_spacing_z_km=2.0)
plt.show()

# Plot a vertical Y-Z cross section
plt.figure(figsize=(9, 9))
pydda.vis.plot_yz_xsection_barbs(Grids, background_field='reflectivity',
                                 level=40,
                                 w_vel_contours=[3, 6, 9, 12, 15],
                                 barb_spacing_y_km=10.0,
                                 barb_spacing_z_km=2.0)
plt.show()
"""
'''

import pyart
import sys
import os
import time

sys.path.append('/Users/clancy/repos/trophy/python/')

#import pydda

#sys.path.append('/Users/clancy/local_packages/jaxpydda')
#import jaxpydda as pydda

#sys.path.append('/Users/clancy/local_packages/dynTRpydda')
sys.path.append('/Users/clancy/repos/trophy/python/dynTRpydda')

sys.path.append('/home/clancy/repos/trophy/python/dynTRpydda')
sys.path.append('/home/clancy/repos/trophy/python/')
import dynTRpydda as pydda

#sys.path.append('/Users/clancy/local_packages/jdynTRpydda')
#import jdynTRpydda as pydda


from matplotlib import pyplot as plt
import numpy
import warnings
warnings.filterwarnings("ignore")


#numpy.show_config()

berr_grid = pyart.io.read_grid(pydda.tests.EXAMPLE_RADAR0)
cpol_grid = pyart.io.read_grid(pydda.tests.EXAMPLE_RADAR1)

sounding = pyart.io.read_arm_sonde(
    pydda.tests.SOUNDING_PATH)


# Load sounding data and insert as an intialization
u_init, v_init, w_init = pydda.initialization.make_wind_field_from_profile(
    cpol_grid, sounding[1], vel_field='corrected_velocity')




# ENTER DESIRED FILE PATH IN QUESTION

tol = 1e-6
subprob_tol = 1e-6
memory_size = 10
prec_vec = [2]
if prec_vec == [1]:
    alg = 'SINGLE_STOL6'
if prec_vec == [2]:
    alg = 'DOUBLE_STOL6'
if prec_vec == [1,2]:
    alg = 'DYNAMIC'
third_folder = '/home/clancy/trophy_data/'+alg+'/'
detail_str = 'memory'+str(memory_size) + '_subprobtol' + "{:.0e}".format(subprob_tol) + '/'
if len(prec_vec)>1:
    detail_str = 'tol' + "{:.0e}".format(tol) + '_' + detail_str

fourth_folder = third_folder + detail_str
if not os.path.isdir(fourth_folder):
    os.system('mkdir ' + fourth_folder)



# Start the wind retrieval. This example only uses the mass continuity
# and data weighting constraints.
ti = time.time()


Grids = pydda.retrieval.get_dd_wind_field([berr_grid, cpol_grid], u_init, v_init, w_init, Co=1.0, Cm=1500.0,
                                          Cz=0, frz=5000.0, filt_iterations=0, mask_outside_opt=True, upper_bc=1,
                                          store_history=True, max_iterations=50000,
                                          max_memory=memory_size, use_dynTR=True, gtol=tol, precision_vector=prec_vec,
                                          subproblem_tol=subprob_tol, write_folder=fourth_folder)



tf = time.time()
print('Time elapsed is', tf-ti)