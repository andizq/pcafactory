import numpy as np
from numpy.ma import masked_array

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
from matplotlib.ticker import ScalarFormatter,FormatStrFormatter
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import matplotlib.text as text
import matplotlib.lines as lines

import os
import sys
import warnings
from collections import OrderedDict
from argparse import ArgumentParser

from astrodendro import Dendrogram
from astropy.io import fits
from astropy import wcs
from utils import get_rt_mode, unit_dict


if int(sys.version[0]) <= 2: 
    message = 'You are using python2.x or below.'
    message += ' Please note that many of the plot adjustments are only'
    message += ' available through Astropy (>=3.x) and therefore python (>=3.x).'
    message += ' This means that your final figure may not be perfect.'
    warnings.warn(message)

parser = ArgumentParser(prog='Peaks in dendrograms', description='Get peaks in dendrogram leaves')
parser.add_argument('-i', '--incl', default='faceon', help="Image inclination ['faceon', 'edgeon', 'edgeon_phi90'] to compute the moment from")
parser.add_argument('-f', '--folder', default='./', help="Location of the input dendrogram")
parser.add_argument('-u', '--unit', default='jypxl', help="Intensity units [jypxl, kelvin, tau]")
parser.add_argument('-R', '--rtmode', default='nonLTE', help="Radiative transfer mode [LTE, nonLTE]")

args = parser.parse_args()
if args.folder[-1] != '/': args.folder += '/'

folder_reg = './portions_moment0/'
print ('creating folder for output files:', folder_reg)
os.system('mkdir %s'%folder_reg)
print (args.incl, args.unit)
#**************************************************
#READING FILES and SETTING CONSTANTS
#**************************************************
tag_tuple = (args.unit,args.incl)

d = Dendrogram.load_from(args.folder+'img_moment0_dendrogram_%s_%s.fits'%tag_tuple)
hdu = fits.open(args.folder+'moment0_img_CO_J1-0_%s_%s_%s.fits'%(args.rtmode, args.unit, args.incl))[0]
w = wcs.WCS(hdu.header)

source_dist = float(np.loadtxt('../pars_size_rt.txt')[1])
pc2au = 206264.806247

#**************************************************
#DENDROGRAM DEFAULT PROPERTIES
#**************************************************
hdu.data *= 1e-3 #m/s to km/s
mean_val = np.mean(hdu.data[hdu.data > 1])

#**************************************************
#CREATING MAIN MASK FOR LEAVES
#**************************************************
mask = np.zeros(hdu.data.shape, dtype=bool)
for leaf in d.leaves:
    mask = mask | leaf.get_mask()

# Now we create a FITS HDU object to contain this, with the correct header
mask_hdu = fits.PrimaryHDU(mask.astype('short'), hdu.header)

#**************************************************
#EXTRACTING ONLY SPECIFIC BRANCHES
#**************************************************
branches = [struct for struct in d.all_structures if (struct.is_branch)]# and struct not in d.trunk)]
minlvl_branches = []
dum, ndesc_list = 0, []

for branch in branches:
    for desc in branch.descendants:
        if desc.is_branch:
            dum += 1
    ndesc_list.append(dum)
    dum = 0

print ('Number of branch-like descendants per branch:', ndesc_list)

sndesc = np.sort(ndesc_list)
if len(sndesc) > 4: wishlvl = [0, sndesc[int(len(ndesc_list)/2 + 2)], sndesc[-2]] #branches with 0 branch-like descendandts, 'median - 2' branch-like descendants and the second biggest branch 
else: wishlvl = ndesc_list

i = 0
for ndesc in ndesc_list:
    if ndesc in wishlvl: minlvl_branches.append(branches[i])
    i+=1

print ('Plotting only those whose number of descendant branches is amongst', wishlvl)

#**************************************************
#CONSTRUCTING SECOND MASK
#**************************************************
num_conts = len(minlvl_branches)
mask_branches = [np.zeros(hdu.data.shape, dtype=bool) for i in range(num_conts)]
i = 0
for branch in minlvl_branches: 
    mask_branches[i] = mask_branches[i] | branch.get_mask()
    i+=1
mask_hdu_bran = [None for i in range(num_conts)]
for i in range(num_conts): mask_hdu_bran[i] = fits.PrimaryHDU(mask_branches[i].astype('short'), hdu.header)

#**************************************************
#PLOTTING 
#**************************************************
#Warning: the wcs projection overwrites the axes matplotlib methods, ax.tick_params is not available in astropy-python2
mpl.rcParams['font.family'] = 'monospace' 
mpl.rcParams['axes.linewidth'] = 1.8
mpl.rcParams['xtick.major.width']=1.3
mpl.rcParams['ytick.major.width']=1.3

SMALL_SIZE = 14
MEDIUM_SIZE = 17
BIGGER_SIZE = 20

fig, ax = plt.subplots(nrows=1, subplot_kw = {'projection': w}, figsize = (8,6))
plt.subplots_adjust(left=0.15, right=0.85, top=0.9, bottom=0.2)
ax.grid()
ax.set_title('Moment 0 map\n'+ r'$^{12}$CO J=1-0', fontsize = BIGGER_SIZE, loc='right')

im = ax.imshow(hdu.data, origin='lower', cmap='pink_r', 
               #vmax = hdu.data.max()/2.
               vmin = 0*mean_val/10.,
               vmax = mean_val + 2*np.std(hdu.data[hdu.data>1])
               )

ax.tick_params(which='both', direction='in', axis='y', 
               labelleft=False, labelright=True, left=True, right=True)
ax.tick_params(axis='both', labelsize=SMALL_SIZE, width=1.3, length=6)

color_dict = dict(colors.BASE_COLORS, **colors.CSS4_COLORS)
color_dict.pop(u'cyan'); color_dict.pop(u'c')
color_names = list(color_dict)

cmp = cm.get_cmap('nipy_spectral')

def plot_colorbar(x0):
    x0 = ax.get_position().x0 + 0.2*x0
    dx = ax.get_position().xmax - 0.085 - x0
    ax_cbar = fig.add_axes([x0,0.07,dx,0.05])
    cbar = fig.colorbar(im, cax=ax_cbar, orientation='horizontal', extend='max')
    cbar.ax.tick_params(labelsize=SMALL_SIZE)
    cbar.set_label(r'M$_0$ (%s km/s)'%unit_dict[args.unit], x = 0.5, fontsize=MEDIUM_SIZE-2)

def plot_leaves():
    leafc = 'red'
    ax.contour(mask_hdu.data, colors=leafc, linewidths=1.0)    
    plt.plot([],[],color=leafc,label='Leaves')

def write_boxes_reg(boxes, file = folder_reg+'clouds_boxes_%s_%s.reg'%tag_tuple):
    f = open(file, 'w')
    header = ['global color=red dash=0 dashlist=8 3 delete=1 edit=1 fixed=0 font="helvetica 10 normal roman" highlite=1 include=1 move=1 select=1 source=1 width=1\n', 'physical\n']
    f.writelines(header)
    boxes_tmp = ['box(%.2f,%.2f,%.2f,%.2f,0)\n'%(x,y,dx,dy) for x,y,dx,dy in boxes]
    f.writelines(boxes_tmp)
    f.close()

def write_peaks_reg(points, file = folder_reg+'clouds_peaks_%s_%s.reg'%tag_tuple):
    f = open(file, 'w')
    header = ['global color=green dash=0 dashlist=8 3 delete=1 edit=1 fixed=0 font="helvetica 10 normal roman" highlite=1 include=1 move=1 select=1 source=1 width=1\n', 'physical\n']
    f.writelines(header)
    points_tmp = ['point(%.2f,%.2f)\n'%(x,y) for x,y in points]
    f.writelines(points_tmp)
    f.close()

def write_boxes_colors(pcolors, file = folder_reg+'clouds_colors_%s_%s.txt'%tag_tuple):
    np.savetxt(file, pcolors, fmt='%s')
    
def write_boxes_rgba(pcolors, file = folder_reg+'clouds_colors_%s_%s.txt'%tag_tuple):
    np.savetxt(file, pcolors)
    
def plot_boxes_peaks(peak_pos, peak_pix):
    width_phys = 30. #pc
    width = width_phys * pc2au #dx pc in au
    height = width_phys * pc2au #dy pc in au

    width_deg = width / source_dist / 3600.   
    height_deg = height / source_dist / 3600. 
    width_pix = abs(width_deg / hdu.header['CDELT1'])
    half_width_pix = 0.5*width_pix
    height_pix = abs(height_deg / hdu.header['CDELT2'])
    half_height_pix = 0.5*height_pix
    boxes, nboxes = [], len(peak_pos)
    box2write, picked_colors, colors_list, rej_border, n = ([], [], cmp(np.linspace(0,1,nboxes)), np.zeros(nboxes).astype(bool), 0)
    for i,(y_deg,x_deg) in enumerate(peak_pos):
        y_pix, x_pix = peak_pix[i]
        pick_color = colors_list[i] #np.random.choice(color_names)
        if ((x_pix - half_width_pix < 0 or x_pix + half_width_pix >= hdu.header['NAXIS1']) or
            (y_pix - half_height_pix < 0 or y_pix + half_height_pix >= hdu.header['NAXIS2'])):
            rej_border[i]=True
            continue
        box2write.append([x_pix,y_pix,width_pix,height_pix])
        picked_colors.append(pick_color)
        rr = Rectangle((x_pix-half_width_pix, y_pix-half_height_pix), 
                       width_pix, height_pix, 
                       linestyle='-', linewidth=2.5, 
                       edgecolor=pick_color, fill=False)
        boxes.append(rr)
        ax.plot([],[],linestyle='-',lw=3,color=pick_color,label='cld %d'%n)
        n+=1
    
    patchcoll = PatchCollection(boxes, match_original=True)
    line = lines.Line2D([0.05, 0.05+2*width_pix/hdu.header['NAXIS1']], [0.04, 0.04],
                        lw=4, color='black', axes=ax, transform=ax.transAxes)
    tt = text.Text(0.05,0.065,'%d pc'%(2*width_phys), fontsize=MEDIUM_SIZE, transform=ax.transAxes)
    ax.add_collection(patchcoll)
    ax.add_line(line)
    ax.add_artist(tt)
    ax.scatter(1*peak_pos[~rej_border,1], 1*peak_pos[~rej_border,0], marker='P', s=70, c='lime', edgecolor='black') #-1 to transpose the arrays, just as the image made by aplpy.
    ax.scatter(1*peak_pos[rej_border,1], 1*peak_pos[rej_border,0], marker='X', s=70, c='lime', edgecolor='black') 
    ax.scatter([],[], marker = 'P', s=70, c='lime', edgecolor='black', label='Peaks')
    if rej_border.any(): ax.scatter([],[], marker = 'X', s=70, c='lime', edgecolor='black', label='Rejected\nborder')

    box2write = np.array(box2write)
    write_boxes_reg(box2write)
    write_peaks_reg(box2write[:,:2])
    #write_boxes_colors(picked_colors)
    write_boxes_rgba(picked_colors)
    return nboxes

def plot_peaks_leaves():
    peak_pos, peak_pix = [], []
    for leaf in d.leaves: 
        y_peak, x_peak = leaf.get_peak()[0]
        peak_pix.append((y_peak,x_peak))
        peak_pos.append((y_peak,x_peak))
    #peak_pos.append(fig.pixel2world(y_peak/hdu.shape[0], x_peak/hdu.shape[1]))
    peak_pos = np.array(peak_pos)
    
    nboxes = plot_boxes_peaks(peak_pos, peak_pix)
    return nboxes

def plot_branches():
    if num_conts > 1: 
        branchc = ['navy'] + ['lightblue'] * (num_conts - 1)
        branchls = [':'] + ['-'] * (num_conts - 1)
    else:
        branchc = ['lightblue']
        branchls = ['-']
    print (branchc)

    for i in range(num_conts): ax.contour(mask_hdu_bran[i].data,
                                          colors=branchc[i],
                                          linestyles = branchls[i],
                                          linewidths=0.8, origin='lower')

    if num_conts > 0:
        ax.plot([],[], branchls[0], color=branchc[0],label='Branch: {bran.idx}'.format(bran = minlvl_branches[0]))
        if num_conts > 1:
            ax.plot([],[],color=branchc[1],
                     label='Branches: {}'.format( ('{bran[%d].idx}, '*(num_conts - 1)%tuple(np.arange(1,num_conts))).format(bran = minlvl_branches))[:-2] )

def plot_legend(nboxes):
    ncols = int(nboxes/16.)+1
    ax.legend(ncol = ncols, framealpha = 0.9, fontsize = SMALL_SIZE, fancybox = True, loc=(-0.39*(ncols-1)-0.35,0))
    ax.text(0.03,0.93, r'$\overline{\rmM}_0$=%.1f'%(mean_val), fontsize=SMALL_SIZE, transform=ax.transAxes)
    return ncols
def plot_dendro():
    nboxes = plot_peaks_leaves(); 
    ncols = plot_legend(nboxes)
    x0 = -ncols*0.275
    plot_colorbar(x0)
    output = "img_moment0_%s_%s.png"%tag_tuple
    plt.savefig(output, dpi = 500, bbox_inches='tight')
    #plot_leaves(); plot_branches(); plot_legend(nboxes)
    #plt.savefig("img_moment0dendro_%s_%s.png"%tag_tuple, dpi = 100, bbox_inches='tight')
    print ('Saving figure on', output)
        
plot_dendro()
#plt.show()
