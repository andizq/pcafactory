import numpy as np
import json
import os
import sys

from astropy.io import fits
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.ticker as ticker
import matplotlib.colors as colors
import matplotlib.patches as mpatches

from utils import literature_dict, alpha_to_gamma

#Three different ways of changing the rcParams on matplotlib:

matplotlib.rcParams['font.family'] = 'monospace'
matplotlib.rcParams['font.weight'] = 'normal'
matplotlib.rcParams['lines.linewidth'] = 1.5
matplotlib.rcParams['axes.linewidth'] = 3.0
matplotlib.rcParams['xtick.major.width']=1.6
matplotlib.rcParams['ytick.major.width']=1.6

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

params = {'xtick.major.size': 6.5,
          'ytick.major.size': 6.5
          }

matplotlib.rcParams.update(params)

incl = ['edgeon', 'faceon', 'edgeon_phi90']
cloudsAB = ['A', 'B']
cloudsCD = ['C', 'D']
time_nSG = ['240', '250']
time_SG = ['235', '240', '245']
time_fdbck = ['253', '260']
cases = ['pot230nSG', 'pot230SGjeans', 'Fjeans248']
rt_folders = ['Lime-arepo', 'Lime-arepo-nonLTE']
units = ['jypxl', 'tau']

root = '/Users/andizq/Downloads/Manchester_2018/Rowan-data/Andres/data/RT-Analysis/'
path = root+'%s/%s_cld%s-Lime/extracted_%s/Subgrids/Dendrograms_portions/portions_moment0/'

def plot_complex(rt_folder, case, cloud, time, unit, incl, num_ticks_x=5, num_ticks_y=4):
    path_base = path%(rt_folder,case,cloud,time)
    path_tuple = (path_base, unit, incl)
    portions = os.popen('ls %s*img*%s_%s_portion*'%path_tuple).readlines()
    nportions = len(portions)

    portion_files = os.popen('ls %s*pca_%s_%s_portion*'%path_tuple).readlines()
    nfiles = len(portion_files)

    img_moment0 = plt.imread('%s../img_moment0_%s_%s.png'%path_tuple)
    colors = np.loadtxt('%sclouds_colors_%s_%s.txt'%path_tuple, dtype=np.float, delimiter=None)
    
    ids = []
    ids_bool = np.zeros(nportions, dtype=np.bool)
    file_overlaped = np.loadtxt('%soverlaped_portions.txt'%path_base, dtype=np.str, delimiter=None)
    if unit=='jypxl': overlaped_portions = np.array(file_overlaped[:,1][file_overlaped[:,0] == incl][0].split(',')).astype(np.int32)
    else: overlaped_portions = np.array(file_overlaped[:,1][file_overlaped[:,0] == incl+unit][0].split(',')).astype(np.int32)

    for file in portion_files:
        name = file.split('\n')[0]
        i = int(name[-7:-4]) #portion_id
        if i in overlaped_portions:
            continue
        ids.append(i)
        ids_bool[i] = True
        
    naccepted = len(ids)
    cube_names = ['%simg_%s_%s'%path_tuple+'_portion_%03d.fits'%id for id in ids]
    cube_data = [fits.getdata(cube)[0] for cube in cube_names]
    cube_header = fits.getheader(cube_names[0])
    
    print (nfiles, nportions, naccepted)
    ncols = 4
    nrows = int(np.ceil(naccepted/4.) )    

    size_inches = (15,5)
    ratio = size_inches[0]/size_inches[1]
    fig = plt.figure(figsize=size_inches)
    ax = [fig.add_axes([0,0,1/3.,1/3.*ratio])]
    ax[0].imshow(img_moment0)
    ax[0].axis('off')

    nchan = cube_header['NAXIS3']
    dchan = cube_header['CDELT3']
    crchan, crval = cube_header['CRPIX3'], cube_header['CRVAL3']
    vel_axis = np.append(np.arange(crval-(crchan-1)*dchan, crval, dchan), np.arange(crval, crval+(nchan-crchan+1)*dchan, dchan))/1000.

    n=0
    for i in range(nrows):
        for j in range(ncols):
            epsx = 3e-3
            epsy = epsx*ratio
            dx = (2/3.-10*epsx)/ncols
            dy = dx*ratio
            ax.append(fig.add_axes([1/3.+(dx+epsx)*j, 1-(dy+epsy)*(i+1.), dx, dx * ratio]))
            n+=1
    
    flux = []
    for i in range(naccepted): flux.append(np.array([np.sum(cube_data[i][ch]) for ch in range(nchan)])/1000.)
    flux_max = np.array(flux).max()

    xlims = (vel_axis[0], vel_axis[-1])
    ylims = (-10, flux_max+10)
    for i in range(naccepted):
        ax[i+1].plot(vel_axis, flux[i], color='k', linewidth=2)
        ax[i+1].fill_between(vel_axis, flux[i], color='0.9') #color=colors[ids[i]], alpha=0.1)#
        ax[i+1].set_ylim(ylims)
        ax[i+1].grid(True, ls=':')
        ax[i+1].set_xticks(np.linspace(*xlims, num_ticks_x))
        ax[i+1].set_yticks(np.linspace(0,ylims[1]-10, num_ticks_y))
        ax[i+1].tick_params(axis="both", direction="in", right=True, labelleft=False, labelbottom=False, labelsize=SMALL_SIZE)        
        for side in ['bottom','top','left','right']: ax[i+1].spines[side].set_color(colors[ids[i]])
        
    for n in range(i+2, nrows*ncols+1): ax[n].axis('off')

    ax[i+1].tick_params(labelright=True, labelbottom=True)
    ax[i+1].set_xlabel(r'$\upsilon_{\rm los}$ (km s$^{-1}$)', fontsize=MEDIUM_SIZE)
    ax[i+1].set_ylabel(r'Flux dens. (kJy)', fontsize=MEDIUM_SIZE-2)
    ax[i+1].yaxis.set_label_position('right')

    plt.savefig('./line_profiles/LINE_%s_cld%s_ext%s_%s_%s.pdf'%(case,cloud,time,unit,incl), dpi=500, bbox_inches='tight')
    return cube_data


def plot_nonLTEvsLTE(case, cloud, time, unit, incl, folder_nonLTE='Lime-arepo-nonLTE', folder_LTE='Lime-arepo', num_ticks_x=5, num_ticks_y=4):
    path_base = path%(folder_nonLTE,case,cloud,time)
    path_tuple = (path_base, unit, incl)
    portions = os.popen('ls %s*img*%s_%s_portion*'%path_tuple).readlines()
    nportions = len(portions)

    portion_files = os.popen('ls %s*pca_%s_%s_portion*'%path_tuple).readlines()
    nfiles = len(portion_files)

    img_moment0 = plt.imread('%s../img_moment0_%s_%s.png'%path_tuple)
    colors = np.loadtxt('%sclouds_colors_%s_%s.txt'%path_tuple, dtype=np.float, delimiter=None)
    
    ids = []
    ids_bool = np.zeros(nportions, dtype=np.bool)
    file_overlaped = np.loadtxt('%soverlaped_portions.txt'%path_base, dtype=np.str, delimiter=None)
    if unit=='jypxl': overlaped_portions = np.array(file_overlaped[:,1][file_overlaped[:,0] == incl][0].split(',')).astype(np.int32)
    else: overlaped_portions = np.array(file_overlaped[:,1][file_overlaped[:,0] == incl+unit][0].split(',')).astype(np.int32)

    for file in portion_files:
        name = file.split('\n')[0]
        i = int(name[-7:-4]) #portion_id
        if i in overlaped_portions:
            continue
        ids.append(i)
        ids_bool[i] = True
        
    naccepted = len(ids)

    path_base_LTE = path%(folder_LTE,case,cloud,time)
    path_tuple_LTE = (path_base_LTE, unit, incl)
    cube_names = [['%simg_%s_%s'%path_tup+'_portion_%03d.fits'%id for id in ids] for path_tup in [path_tuple, path_tuple_LTE]]
    cube_data = [[fits.getdata(cube)[0] for cube in cube_names[i]] for i in range(2)]
    cube_header = fits.getheader(cube_names[0][0])
    
    print (nfiles, nportions, naccepted)
    ncols = 4
    nrows = int(np.ceil(naccepted/4.) )    

    size_inches = (15,5)
    ratio = size_inches[0]/size_inches[1]
    fig = plt.figure(figsize=size_inches)
    ax = [fig.add_axes([0,0,1/3.,1/3.*ratio])]
    ax[0].imshow(img_moment0)
    ax[0].axis('off')

    nchan = cube_header['NAXIS3']
    dchan = cube_header['CDELT3']
    crchan, crval = cube_header['CRPIX3'], cube_header['CRVAL3']
    vel_axis = np.append(np.arange(crval-(crchan-1)*dchan, crval, dchan), np.arange(crval, crval+(nchan-crchan+1)*dchan, dchan))/1000.

    n=0
    for i in range(nrows):
        for j in range(ncols):
            epsx = 3e-3
            epsy = epsx*ratio
            dx = (2/3.-10*epsx)/ncols
            dy = dx*ratio
            ax.append(fig.add_axes([1/3.+(dx+epsx)*j, 1-(dy+epsy)*(i+1.), dx, dx * ratio]))
            n+=1
    
    flux = [[], []]
    for n in range(2):
        for i in range(naccepted): 
            flux[n].append(np.array([np.sum(cube_data[n][i][ch]) for ch in range(nchan)])/1000.)
    flux_max = np.array(flux[1]).max() #LTE is usually higher

    for n in range(2):
        for i in range(naccepted):
            if n==0:
                ax[i+1].plot(vel_axis, flux[n][i], color='k', linewidth=2.0)
                ax[i+1].fill_between(vel_axis, flux[n][i], color='0.9') #color=colors[ids[i]], alpha=0.1)#
            if n==1: ax[i+1].plot(vel_axis, flux[n][i], color='k', linestyle='-.', linewidth=1.0)

    xlims = (vel_axis[0], vel_axis[-1])
    ylims = (-10, flux_max+10)
    for i in range(naccepted):
        ax[i+1].set_ylim(ylims)
        ax[i+1].grid(True, ls=':')
        ax[i+1].set_xticks(np.linspace(*xlims, num_ticks_x))
        ax[i+1].set_yticks(np.linspace(0,ylims[1]-10, num_ticks_y))
        ax[i+1].tick_params(axis="both", direction="in", right=True, labelleft=False, labelbottom=False, labelsize=SMALL_SIZE)        
        for side in ['bottom','top','left','right']: ax[i+1].spines[side].set_color(colors[ids[i]])
        
    for n in range(i+2, nrows*ncols+1): ax[n].axis('off')

    ax[i+1].tick_params(labelright=True, labelbottom=True)
    ax[i+1].set_xlabel(r'$\upsilon_{\rm los}$ (km s$^{-1}$)', fontsize=MEDIUM_SIZE)
    ax[i+1].set_ylabel(r'Flux dens. (kJy)', fontsize=MEDIUM_SIZE-2)
    ax[i+1].yaxis.set_label_position('right')

    ax[1].plot([None],[None], color='k', linestyle='-.', linewidth=1.0, label='LTE')
    ax[1].plot([None],[None], color='k', linestyle='-', linewidth=2.0, label='non-LTE')
    ax[1].legend(fontsize=SMALL_SIZE, loc='upper left')

    plt.savefig('./line_profiles/LINE_nonLTEvsLTE_%s_cld%s_ext%s_%s_%s.pdf'%(case,cloud,time,unit,incl), dpi=500, bbox_inches='tight')
    return cube_data
    
#cube_data = plot_complex(rt_folders[0],cases[2],cloudsCD[1],time_fdbck[0],units[0],incl[2])
#cube_data = plot_complex(rt_folders[0],cases[1],cloudsAB[1],time_SG[2],units[0],incl[2])
cube_data = plot_nonLTEvsLTE(cases[0],cloudsAB[1],time_nSG[0],units[0],incl[2])
cube_data = plot_nonLTEvsLTE(cases[0],cloudsAB[1],time_nSG[0],units[0],incl[0])
cube_data = plot_nonLTEvsLTE(cases[0],cloudsAB[1],time_nSG[0],units[0],incl[1])
cube_data = plot_nonLTEvsLTE(cases[1],cloudsAB[1],time_SG[1],units[0],incl[2])
cube_data = plot_nonLTEvsLTE(cases[2],cloudsCD[1],time_fdbck[0],units[0],incl[2])

plt.show()

#Missing: plot line profile from the whole complex
#Missing: overlaid nonLTE profiles
