import numpy as np
import sys
import os
import time
from functools import reduce

from arepy.read_write import binary_read as rsnap

import sf3dmodels.Model as Model
from sf3dmodels import Plot_model
from sf3dmodels.utils.units import pc
from sf3dmodels.arepo.units import *
import sf3dmodels.grid as grid

from argparse import ArgumentParser

parser = ArgumentParser(prog='Plot projected properties', description='Plots the 2D projection of a given set of properties')
parser.add_argument('-g', '--gridoverlap', default=1, type=int, 
                    help="If 1 (True), reads the properties from file and overlaps them into a global grid. If 0 (False), it only performs the plotting part, so it will work only if the overlaping was already run and saved in memory (for instance using ipython: run -i plot_column.py)")
parser.add_argument('-i', '--incl', default='faceon', help="Model image inclination ['edgeon' or 'faceon'] to use")
#parser.add_argument('-f', '--folder', default='../', help="Folder location of data cube")
args = parser.parse_args()

t0 = time.time()

def get_cloud_history(add=None):
    cwd = os.getcwd()
    arepo_step = 0.2 #Myr    
    cloud = cwd.split('cld')[1][0]
    case_dict = {'Fjeans': 'Self-Gravity + Mixed Supernovae', 
                 'nSG': 'No Self-Gravity + Random Supernovae', 
                 'SGjeans': 'Self-Gravity + Random Supernovae'}
    for key in case_dict: 
        if key in cwd: case = case_dict[key]; break
    
    if key == 'Fjeans': pot_time = int(cwd.split(key)[1][0:3])
    else: pot_time = int(cwd.split(key)[0][-3:])
    cloud_time = int(cwd.split('extracted_')[1][0:3])
    time = (cloud_time - pot_time) * arepo_step
    extras = ''
    if add != None:
        for a in add: extras+= ' $-$ %s'%a
    return r'%s, cloud %s, %.1f Myr old'%(case, cloud, time) + extras

def do_overlap():
    #******************************
    #PATH AND TAG
    #******************************
    cwd = os.getcwd()
    cloud_kind = cwd.split('Lime-arepo/')[-1].split('-Lime')
    cloud_kind = cloud_kind[0] + cloud_kind[1]
    home = os.environ['HOME']
    base = home+'/Downloads/Manchester_2018/Rowan-data/Andres/data/datasets/'
    base += cloud_kind 
    base_content = os.popen('ls '+base).readlines()
    snapshot = ''
    for file in base_content:
        if 'Restest' in file: snapshot = '/'+file.split('\n')[0]

    #******************************
    #AREPO 3D-grids (not projected)
    #******************************
    rsnap.io_flags['mc_tracer']=True
    rsnap.io_flags['time_steps']=True
    rsnap.io_flags['sgchem']=True
    rsnap.io_flags['variable_metallicity']=False
    #rsnap.io_flags['MHD']=True #There is no magnetic field information
    #******************************
    #READING SNAPSHOT
    #******************************
    f = base + snapshot 
    data, header = rsnap.read_snapshot(f)
    nparts = header['num_particles']
    ngas = nparts[0]
    nsinks = nparts[5]
    for key in data: 
        print(key, np.shape(data[key]))
    #****************************
    #GLOBAL GRID
    #****************************
    pos_max = np.max(data['pos'], axis=0)
    pos_min = np.min(data['pos'], axis=0)
    pos_mean = 0.5 * (pos_max + pos_min) * ulength / 100.

    sizex = sizey = sizez = 0.5*np.max(pos_max-pos_min) * ulength / 100.
    #print (sizex, sizey, sizez)
    Nx = Ny = Nz = 250
    GRID = Model.grid([sizex,sizey,sizez], [Nx,Ny,Nz], include_zero=False)#, rt_code='radmc3d')    
    columns = ['id', 'x', 'y', 'z', 'dens_H2', 'dens_H', 'dens_Hplus', 'temp_gas',
               'temp_dust', 'vel_x', 'vel_y', 'vel_z', 'abundance', 'gtdratio',
               'doppler']
    overlap = grid.Overlap(GRID)
    overlap.min_values['dens_H2'] = 0.0
    newprop = overlap.fromfiles(columns, submodels=['datatab.dat'])
    
    sink_pos = data['pos'][ngas:] * ulength / 100 - pos_mean
    return GRID, sizex, newprop, sink_pos, nsinks

#**********************
#PREAMBLE
#**********************
if args.gridoverlap: 
    GRID, sizex, newprop, sink_pos, nsinks = do_overlap()
    prop3d = {key: np.reshape(newprop[key], GRID.Nodes, order='C') for key in newprop} #Third index changes fastest (lime)
    for key_dens in ['dens_H', 'dens_H2', 'dens_Hplus']:
        prop_tmp = prop3d[key_dens]*1e-4 #1e-4 to get cm**-2 from m**-2
        prop3d.update({key_dens: prop_tmp}) #update() instead of prop*=1e-4 to avoid repeat the operation when using the ipython memory via run -i
  
Nx,Ny,Nz = GRID.Nodes
xyz = GRID.XYZgrid

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
import matplotlib.cm as cm
import matplotlib.ticker as ticker
import matplotlib.colors as colors
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches
#Three different ways to change the rcParams on matplotlib:

matplotlib.rcParams['font.family'] = 'monospace'
matplotlib.rcParams['lines.linewidth'] = 1.5
matplotlib.rcParams['axes.linewidth'] = 1.8
matplotlib.rcParams['xtick.major.width']=1.3
matplotlib.rcParams['ytick.major.width']=1.3

SMALL_SIZE = 10
MEDIUM_SIZE = 15
BIGGER_SIZE = 22

matplotlib.rc('font', size=BIGGER_SIZE)          # controls default text sizes
matplotlib.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
matplotlib.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
matplotlib.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
matplotlib.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
matplotlib.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
matplotlib.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

params = {'xtick.major.size': 5.5,
          'ytick.major.size': 5.5
          }

matplotlib.rcParams.update(params)

def masked_prop(prop, factor=1., wfunc=np.mean, val=None):
    if val is None: val = factor*wfunc(prop)
    return np.ma.masked_where(prop<=val, prop)

def sort_orientation(prop, incl):
    if incl == 'faceon': return prop.T
    elif incl == 'edgeon': return prop.T #[::-1]
    elif incl == 'edgeon_phi90': return prop.T #[::-1]

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=256):
    cmap_name = 'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
    new_cmap = colors.LinearSegmentedColormap.from_list(cmap_name,
                                                        cmap(np.linspace(minval, maxval, n)), 
                                                        N=n)
    return new_cmap

#*********************
#PLOTTING
#*********************
axis_dict = {'edgeon_phi90': 0, 'edgeon': 1, 'faceon': 2}
axis_los = axis_dict[args.incl] #line of sight axis
axis_id = np.arange(3)
axis_id = axis_id[axis_id != axis_los]

size_inches = (11.,6.)
#fig, ax = plt.subplots(nrows=1, figsize=size_inches)
fig = plt.figure(figsize=size_inches)

ax = fig.add_axes([0.085,0.15,0.4,(0.4)*size_inches[0]/size_inches[1]])
#fig.subplots_adjust(left=0., bottom=0.15, right=0.5, top=0.85, wspace=0.0, hspace=0.0)

c0 = sizex/pc #np.max(np.abs([xyz[axis_id[0]][0], xyz[axis_id[1]][0], xyz[axis_id[0]][-1], xyz[axis_id[1]][-1]]))/pc
plot_extent = [-c0,c0,-c0,c0]

#*********************
#ATOMIC HYDROGEN
#*********************
#plot_T = sort_orientation(np.sum(prop3d['dens_H']*prop3d['temp_gas'], axis=axis_los)/np.sum(prop3d['dens_H'], axis=axis_los), args.incl) * GRID.step[axis_los]

sink_plot = ax.scatter(*sink_pos.T[axis_id]/pc, s=40, marker='*', linewidth=0.8, facecolor='w', edgecolor='k')
ax.text(0.02,0.94, r'N$_{\rm sinks}$=%d'%nsinks, color='w', fontsize=MEDIUM_SIZE, transform=ax.transAxes)

plot_H = sort_orientation(np.sum(prop3d['dens_H'], axis=axis_los), args.incl) * GRID.step[axis_los]
norm_H = LogNorm(vmin=0.1*np.mean(plot_H), vmax=np.max(plot_H))
img_H = ax.imshow(plot_H, cmap='plasma', norm=norm_H, origin='lower', extent=plot_extent)

rt_radius = np.loadtxt('./Subgrids/pars_size_rt.txt')[0]
lime_domain = mpatches.Circle((0,0), radius=rt_radius, ls='-', lw=2.5, ec='white', fc='none')
ax.add_patch(lime_domain)

#***********************
#MOLECULAR HYDROGEN 
#***********************
ax1 = fig.add_axes([0.61,0.25,0.31,0.31*reduce(lambda x,y: x/y, size_inches)]) #0.615

plot_H2 = sort_orientation(np.sum(prop3d['dens_H2'], axis=axis_los), args.incl) * GRID.step[axis_los]
norm_H2 = LogNorm(vmin=0.1*np.mean(plot_H2), vmax=1*np.max(plot_H2))
masked_H2 = masked_prop(plot_H2, factor=0.01)
img_H2 = ax1.imshow(masked_H2, cmap='hot', norm=norm_H2, origin='lower', extent=plot_extent)

#***********************
#CO ABUNDANCE
#***********************
num_levels = 5
mol_dens = prop3d['abundance'] * (prop3d['dens_H'] + 2*prop3d['dens_H2'] + prop3d['dens_Hplus'])
plot_abund = sort_orientation(np.sum(mol_dens, axis=axis_los), args.incl) * GRID.step[axis_los]
levels_contour = np.linspace(*np.log10([1*np.mean(plot_abund), 1*np.max(plot_abund)]), num=num_levels)
hdelta = 0.5*(levels_contour[1] - levels_contour[0])
boundaries = np.linspace(levels_contour[0]-hdelta, levels_contour[-1]+hdelta, num=num_levels+1)

norm_abund = BoundaryNorm(boundaries, num_levels)
cmap_contour = plt.get_cmap("Greens")
cmap_contour = truncate_colormap(cmap_contour, 0.3, 1.0, n=num_levels)

def fmt_func(x,pos):
    level_exp = levels_contour[norm_abund(x)]
    coeff = round(10**(level_exp % 1)) #10**(decimal part)
    return r'%d$\times$10$^{%d}$'%(coeff,level_exp)
fmt = ticker.FuncFormatter(fmt_func)
ab = ax1.contour(np.log10(plot_abund), cmap=cmap_contour, norm=norm_abund, levels = levels_contour, extent=plot_extent)

xdum, ydum = [[None]*num_levels]*2 
sc = ax1.scatter(xdum,ydum, c=levels_contour, norm=norm_abund, cmap=cmap_contour)

lime_domain = mpatches.Circle((0,0), radius=rt_radius, ls='--', lw=1.5, ec='k', fc='none', transform = ax1.transData)
img_H2.set_clip_path(lime_domain)
ax1.add_patch(lime_domain)


#*****************
#COLORBARS
#*****************
ax0_pos = ax.get_position()
ax1_pos = ax1.get_position()
ax0_cbar0 = fig.add_axes([ax0_pos.x0+0.0705,ax0_pos.y1,0.33,0.05]) 
ax1_cbar0 = fig.add_axes([ax1_pos.x0-0.015,ax1_pos.y1+0.02,0.33,0.05]) 
ax1_cbar1 = fig.add_axes([ax1_pos.x0-0.015,ax1_pos.y0-0.06,0.33,0.05]) 
cbar00=fig.colorbar(img_H, cax=ax0_cbar0, extend='min', orientation='horizontal')
cbar10=fig.colorbar(img_H2, cax=ax1_cbar0, extend='min', orientation='horizontal')
cbar11=fig.colorbar(sc, cax=ax1_cbar1, extend='min', ticks=levels_contour, orientation='horizontal', format=fmt)
cbar00.ax.tick_params(axis='x', which='both', direction='out', top=True, bottom=False, labeltop=True, labelbottom=False)
cbar10.ax.tick_params(axis='x', which='both', direction='out', top=True, bottom=False, labeltop=True, labelbottom=False)
cbar11.ax.tick_params(axis='x', which='both', direction='out', top=False, bottom=True, labeltop=False, labelbottom=True, labelrotation=45, labelsize=12)
cbar00.set_label(r'$\Sigma_{\rm H}$')#, labelpad=-45, x=0)
cbar10.set_label(r'$\Sigma_{\rm H_2}$')#, labelpad=10, x=0)
cbar11.set_label(r'$\Sigma_{\rm CO}$'+'\n[cm$^{-2}]$')#, labelpad=-90, x=0)
cbar00.ax.xaxis.set_label_coords(-0.08, 1.0)
cbar10.ax.xaxis.set_label_coords(-0.08, 1.0)
cbar11.ax.xaxis.set_label_coords(-0.08, 1.0)

ax_labels = ['x','y','z']
ax.set_xlabel(ax_labels[axis_id[0]] + '/pc')
ax.set_ylabel(ax_labels[axis_id[1]])
ax1.text(1.15,1.20, get_cloud_history(add=[args.incl])+r' $-$ RT radius %d pc'%rt_radius, fontsize=SMALL_SIZE-2.7, transform=ax1.transAxes, rotation=-90)
#ax1.set_facecolor('k')
ax1.axis('off')
for spine in ax1.spines: ax1.spines[spine].set_visible(False) #Turning off the spines does not delete axis labels
ax1.tick_params(axis='both', left=False, bottom=False)
fig.savefig('column_densities_%s.pdf'%args.incl, bbox_inches=None, dpi=500)
plt.show()
#-------
#TIMING
#-------
print ('Ellapsed time: %.3fs' % (time.time() - t0))
print ('-------------------------------------------------\n-------------------------------------------------\n')

#RUNNING IT:
#ipython3
#file = %env TOOLS_PCA 
#file += 'plot_column.py'
#run -i {file} -g 1 #to make the 3D grid and save it into memory
#run -i {file} -g 0 #to only plot the figure
