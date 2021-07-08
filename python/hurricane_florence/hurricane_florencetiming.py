import urllib
import pyart
import sys, os
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np

# change to location of trophy repo
repo_path = os.environ['HOME'] + '/repos/'
sys.path.insert(0, repo_path + 'trophy/python/')
sys.path.insert(0, repo_path + 'trophy/python/dynTRpydda_for_timing/')
import dynTRpydda_for_timing as pydda

warnings.filterwarnings("ignore")

#hrrr_url = ('https://pando-rgw01.chpc.utah.edu/hrrr/prs/20180914/' +
#            'hrrr.t06z.wrfprsf00.grib2')
#urllib.request.urlretrieve(hrrr_url, 'test.grib2')
#precisions = {'half': 0, 'single': 1, 'double': 2}
precisions = {'single': 1}#, 'double': 2}


tol = 1e-2
subprob_tol = 1e-6
memory_size = 0
seed_val = 1
dt_string = 'run' + str(seed_val)
alg = ''
for key in precisions.keys():
    alg += '_' + key
alg = alg[1::]
if os.path.isdir('/home/clancy'):
    third_folder = '/home/clancy/trophy_data/hurricane_florence/'+alg+'/'
if os.path.isdir('/home/rclancy'):
    third_folder = '/home/rclancy/trophy_data/hurricane_florence/'+alg+'/'
if os.path.isdir('/Users/clancy'):
    third_folder = '/Users/clancy/trophy_data/hurricane_florence/'+alg+'/'
if not os.path.isdir(third_folder):
    os.system('mkdir ' + third_folder)
detail_str = 'memory'+str(memory_size) + '_subprobtol' + "{:.0e}".format(subprob_tol) + '/'
if len(precisions) > 1:
    detail_str = 'tol' + "{:.0e}".format(tol) + '_' + detail_str
fourth_folder = third_folder + detail_str
if not os.path.isdir(fourth_folder):
    os.system('mkdir ' + fourth_folder)
fourth_folder += dt_string + '/'
if not os.path.isdir(fourth_folder):
    os.system('mkdir ' + fourth_folder)



grid_mhx = pyart.io.read_grid(u'grid_mhx.nc')
grid_ltx = pyart.io.read_grid(u'grid_ltx.nc')

grid_mhx = pydda.constraints.add_hrrr_constraint_to_grid(grid_mhx,
                                                         'test.grib2')
u_init, v_init, w_init = pydda.initialization.make_constant_wind_field(
    grid_mhx, (0.0, 0.0, 0.0))

# ADDED BY RICHIE FOR WARM START
# get shape of winds 3d grid
aa, bb, cc = u_init.shape

# read warm start, extract winds column, then convert to numpy array and reshape
#init_winds = np.asarray(pd.read_csv('./winds_and_gradient.csv')['winds'])
#init_winds = np.reshape(init_winds, (3, aa, bb, cc))
"""
u_init = init_winds[0]
v_init = init_winds[1]
w_init = init_winds[2]

u_init = np.zeros(u_init.shape)
v_init = np.zeros(u_init.shape)
w_init = np.zeros(u_init.shape)
"""


np.random.seed(seed_val)

u_init = u_init + np.random.normal(0, 1, u_init.shape)
v_init = v_init + np.random.normal(0, 1, v_init.shape)
w_init = w_init + np.random.normal(0, 1, w_init.shape)



out_grids = pydda.retrieval.get_dd_wind_field(
    [grid_mhx, grid_ltx], u_init, v_init, w_init, Co=0.1, Cm=1000.0, Cmod=1e-3,
    mask_outside_opt=True, vel_name='corrected_velocity', model_fields=["hrrr"], store_history=True,
    filt_iterations=0, max_memory=memory_size, use_dynTR=True, gtol=tol, precision_dict=precisions,
    subproblem_tol=subprob_tol, write_folder=fourth_folder)

