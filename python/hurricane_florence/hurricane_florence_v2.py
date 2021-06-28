import urllib
import pyart
import sys, os

#import pydda

#sys.path.append('/Users/clancy/local_packages/jaxpydda')
#import jaxpydda as pydda

#sys.path.append('/Users/clancy/local_packages/')
#sys.path.append('/Users/clancy/local_packages/dynTRpydda')
sys.path.append('/Users/clancy/repos/trophy/python/')
sys.path.append('/Users/clancy/repos/trophy/python/dynTRpydda_edits')
import dynTRpydda_edits as pydda


import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np

#hrrr_url = ('https://pando-rgw01.chpc.utah.edu/hrrr/prs/20180914/' +
#            'hrrr.t06z.wrfprsf00.grib2')
#urllib.request.urlretrieve(hrrr_url, 'test.grib2')
#precisions = {'half': 0, 'single': 1, 'double': 2}
precisions = {'single': 1, 'double': 2}


tol = 1e0
subprob_tol = 1e-7
memory_size = 10
dt_string = 'run' + str("test2")
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
out_grids = pydda.retrieval.get_dd_wind_field(
    [grid_mhx, grid_ltx], u_init, v_init, w_init, Co=0.1, Cm=1000.0, Cmod=1e-3,
    mask_outside_opt=True, vel_name='corrected_velocity', model_fields=["hrrr"], store_history=True,
    filt_iterations=0, max_memory=memory_size, use_dynTR=True, gtol=tol, precision_dict=precisions,
    subproblem_tol=subprob_tol, write_folder=fourth_folder)

"""
Grids = pydda.retrieval.get_dd_wind_field([berr_grid, cpol_grid], u_init, v_init, w_init, Co=1.0, Cm=1500.0,
                                          Cz=0, frz=5000.0, filt_iterations=0, mask_outside_opt=True, upper_bc=1,
                                          store_history=True, max_iterations=50000,
                                          max_memory=memory_size, use_dynTR=True, gtol=tol, precision_dict=precisions,
                                          subproblem_tol=subprob_tol, write_folder=fourth_folder)
"""



fig = plt.figure(figsize=(25, 15))
ax = plt.axes(projection=ccrs.PlateCarree())
ax = pydda.vis.plot_horiz_xsection_barbs_map(
    out_grids, ax=ax, bg_grid_no=-1, level=1, barb_spacing_x_km=20.0,
    barb_spacing_y_km=20.0, cmap='pyart_HomeyerRainbow')
ax.set_xticks(np.arange(-80, -75, 0.5))
ax.set_yticks(np.arange(33., 35.5, 0.5))
plt.title(out_grids[0].time['units'][13:] + ' winds at 0.5 km')
plt.show()