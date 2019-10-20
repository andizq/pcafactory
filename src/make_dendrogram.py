import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from astrodendro import Dendrogram
from astropy.io import fits
from astropy import wcs
from utils import get_rt_mode, unit_dict

parser = ArgumentParser(prog='Dendrogram on moment 0', description='Compute dendrogram onn moment 0 from data cube')
parser.add_argument('-i', '--incl', default='faceon', help="Image inclination ['faceon', 'edgeon', 'edgeon_phi90'] to compute the moment from")
parser.add_argument('-f', '--folder', default='./', help="Location of the input moment0")
parser.add_argument('-u', '--unit', default='jypxl', help="Intensity units [jypxl, kelvin, tau]")
parser.add_argument('-R', '--rtmode', default='nonLTE', help="Radiative transfer mode [LTE, nonLTE]")

args = parser.parse_args()
if args.folder[-1] != '/': args.folder += '/'
#**************************************************
#READING FILES
#**************************************************
hdu = fits.open(args.folder+'moment0_img_CO_J1-0_%s_%s_%s.fits'%(args.rtmode, args.unit, args.incl))[0]
#hdu.header['CUNIT3'] = 'm/s'
pars = np.loadtxt('pars_dendrogram.txt', dtype=np.str, delimiter=None, unpack=True) 
if args.unit == 'jypxl': id = pars[0] == args.incl 
else: id = pars[0] == args.incl+args.unit
#**************************************************
#COMPUTING DENDROGRAM
#**************************************************
hdu.data *= 1e-3
mean_val = hdu.data.mean() 
min_val = hdu.data.mean()/10
if pars.ndim==1: min_delta = float(pars[1])*mean_val; min_npix=int(pars[2])
else: min_delta = float(pars[1][id])*mean_val; min_npix=int(pars[2][id])

d = Dendrogram.compute(hdu.data.squeeze(), min_value=min_val, min_delta=min_delta, min_npix=min_npix)
d.save_to('img_moment0_dendrogram_%s_%s.fits'%(args.unit,args.incl))
w = wcs.WCS(hdu.header)

#**************************************************
#CREATING MAIN MASK FOR LEAVES
#**************************************************
mask = np.zeros(hdu.data.shape, dtype=bool)
for leaf in d.leaves:
    mask = mask | leaf.get_mask()

print ('Number of leaves:', len(d.leaves))
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


#*************************
#PLOTTING DENDROGRAM on M0
#*************************
mpl.rcParams['font.family'] = 'monospace' 
mpl.rcParams['axes.linewidth'] = 1.8
mpl.rcParams['xtick.major.width']=1.3
mpl.rcParams['ytick.major.width']=1.3

TINY_SIZE = 10
SMALL_SIZE = 14
MEDIUM_SIZE = 17
BIGGER_SIZE = 20

def plot_main():
    fig, ax = plt.subplots(nrows=1, subplot_kw = {'projection': w}, figsize = (8,6))
    plt.subplots_adjust(left=0.15, right=0.85, top=0.9, bottom=0.2)
    ax.grid()
    ax.set_title('Moment 0 map\n'+ r'$^{12}$CO J=1-0', fontsize = BIGGER_SIZE, loc='right')

    im = ax.imshow(hdu.data, origin='lower', cmap='viridis_r', 
                   vmin = 0*mean_val/10.,
                   vmax = np.mean(hdu.data[hdu.data > 1]) + 2*np.std(hdu.data[hdu.data>1])
                   )

    ax.tick_params(which='both', direction='in', axis='y', 
                   labelleft=False, labelright=True, left=True, right=True)
    ax.tick_params(axis='both', labelsize=SMALL_SIZE, width=1.3, length=6)
    return fig, ax, im

def plot_colorbar(fig, ax, im, x0):
    x0 = ax.get_position().x0 + 0.2*x0
    dx = ax.get_position().xmax - 0.085 - x0
    ax_cbar = fig.add_axes([x0,0.07,dx,0.05])
    cbar = fig.colorbar(im, cax=ax_cbar, orientation='horizontal', extend='max')
    cbar.ax.tick_params(labelsize=SMALL_SIZE)
    cbar.set_label(r'M$_0$ (%s km/s)'%unit_dict[args.unit], x = 0.5, fontsize=MEDIUM_SIZE-2)

def plot_leaves(ax):
    leafc = 'red'
    ax.contour(mask_hdu.data, colors=leafc, linewidths=1.0)    
    ax.plot([],[],color=leafc,label='Leaves')

def plot_branches(ax):
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
        ax.plot([],[], branchls[0], color=branchc[0],label='Branch id: {bran.idx}'.format(bran = minlvl_branches[0]))
        if num_conts > 1:
            ax.plot([],[],color=branchc[1],
                     label='Branch ids: {}'.format( ('\n{bran[%d].idx}, '*(num_conts - 1) % tuple(np.arange(1,num_conts))).format(bran = minlvl_branches))[:-2] )

def plot_legend(ax):
    ax.legend(loc=(-0.42,0.02), framealpha = 0.97, fontsize = TINY_SIZE+1, fancybox = True)

def plot_dendro():
    fig, ax, im = plot_main()
    x0 = -0.275
    plot_colorbar(fig, ax, im, x0)
    plot_leaves(ax); plot_branches(ax); plot_legend(ax)
    output = "img_moment0dendro_%s_%s.png"%(args.unit,args.incl)
    plt.savefig(output, dpi=100, bbox_inches='tight')
    print ('Saving figure on', output)

plot_dendro()

