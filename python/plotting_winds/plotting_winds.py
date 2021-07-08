import pyart
import sys
import time
from matplotlib import pyplot as plt
import numpy
import warnings

# change to location of trophy repo
repo_path = os.environ['HOME'] + '/repos/'
sys.path.append(repo_path + 'trophy/python/')
sys.path.append(repo_path + 'trophy/python/dynTRpydda/')

import dynTRpydda as pydda

warnings.filterwarnings("ignore")

berr_grid = pyart.io.read_grid(pydda.tests.EXAMPLE_RADAR0)
cpol_grid = pyart.io.read_grid(pydda.tests.EXAMPLE_RADAR1)

sounding = pyart.io.read_arm_sonde(
    pydda.tests.SOUNDING_PATH)


# Load sounding data and insert as an intialization
u_init, v_init, w_init = pydda.initialization.make_wind_field_from_profile(
    cpol_grid, sounding[1], vel_field='corrected_velocity')

# Start the wind retrieval. This example only uses the mass continuity
# and data weighting constraints.
ti = time.time()

# use trophy solver with jax calls
Grids = pydda.retrieval.get_dd_wind_field([berr_grid, cpol_grid], u_init, v_init, w_init, Co=1.0, Cm=1500.0,
                                          Cz=0, frz=5000.0, filt_iterations=2, mask_outside_opt=True, upper_bc=1,
                                          store_history=True, max_iterations=10000,
                                          max_memory=10, use_dynTR=True, gtol=1e-1)

"""
AVAILABLE OPTIONS
def get_dd_wind_field(Grids, u_init, v_init, w_init, points=None, vel_name=None,
                      refl_field=None, u_back=None, v_back=None, z_back=None,
                      frz=4500.0, Co=1.0, Cm=1500.0, Cx=0.0,
                      Cy=0.0, Cz=0.0, Cb=0.0, Cv=0.0, Cmod=0.0, Cpoint=0.0,
                      Ut=None, Vt=None, filt_iterations=2,
                      mask_outside_opt=False, weights_obs=None,
                      weights_model=None, weights_bg=None,
                      max_iterations=10000, mask_w_outside_opt=True,
                      filter_window=9, filter_order=3, min_bca=30.0,
                      max_bca=150.0, upper_bc=True, model_fields=None,
                      output_cost_functions=True, roi=1000.0,
                      max_memory=10, use_dynTR=True, store_history=False):
"""

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