import urllib
import pyart
import sys
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np


# change to location of trophy repo
repo_path = os.environ['HOME'] + '/repos/'
sys.path.insert(0, repo_path + 'trophy/python/')
sys.path.insert(0, repo_path + 'trophy/python/dynTRpydda/')

import dynTRpydda as pydda



#hrrr_url = ('https://pando-rgw01.chpc.utah.edu/hrrr/prs/20180914/' +
#            'hrrr.t06z.wrfprsf00.grib2')
#urllib.request.urlretrieve(hrrr_url, 'test.grib2')

grid_mhx = pyart.io.read_grid(u'grid_mhx.nc')
grid_ltx = pyart.io.read_grid(u'grid_ltx.nc')

grid_mhx = pydda.constraints.add_hrrr_constraint_to_grid(grid_mhx,
                                                         'test.grib2')
u_init, v_init, w_init = pydda.initialization.make_constant_wind_field(
    grid_mhx, (0.0, 0.0, 0.0))
out_grids = pydda.retrieval.get_dd_wind_field(
    [grid_mhx, grid_ltx], u_init, v_init, w_init, Co=0.1, Cm=1000.0, Cmod=1e-3,
    mask_outside_opt=True, vel_name='corrected_velocity',
    model_fields=["hrrr"])

fig = plt.figure(figsize=(25, 15))
ax = plt.axes(projection=ccrs.PlateCarree())
ax = pydda.vis.plot_horiz_xsection_barbs_map(
    out_grids, ax=ax, bg_grid_no=-1, level=1, barb_spacing_x_km=20.0,
    barb_spacing_y_km=20.0, cmap='pyart_HomeyerRainbow')
ax.set_xticks(np.arange(-80, -75, 0.5))
ax.set_yticks(np.arange(33., 35.5, 0.5))
plt.title(out_grids[0].time['units'][13:] + ' winds at 0.5 km')
plt.show()