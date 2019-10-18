import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from astrodendro import Dendrogram
from astropy.io import fits
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
print (pars[0][id])
#**************************************************
#COMPUTING DENDROGRAM
#**************************************************
hdu.data *= 1e-3
mean_val = hdu.data.mean() 
min_val = hdu.data.mean()/10
d = Dendrogram.compute(hdu.data.squeeze(), min_value=min_val, min_delta=float(pars[1][id])*mean_val, min_npix=int(pars[2][id]))
d.save_to('img_moment0_dendrogram_%s_%s.fits'%(args.unit,args.incl))

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

#**************************************************
#PLOTTING by APLpy
#**************************************************

try: import aplpy
except ImportError:
    raise ImportError("The dendrograms were successfully saved and you can carry on with your analysis. However, the dendrograms visualisation failed because you don't have the aplpy package installed (https://aplpy.readthedocs.io)")
    
fig = aplpy.FITSFigure(hdu, figsize=(8, 6))
fig.add_label(0.5, 1.10, 'Moment 0\nFrom $^{12}CO$ $J=1-0$', relative=True, size = 12)
fig.show_colorscale(cmap='viridis_r', stretch='linear',
                    vmin = 0*mean_val/10.,
                    vmax = mean_val + 2*np.std(hdu.data[hdu.data>1])
                    )
fig.add_colorbar()
fig.colorbar.set_axis_label_text(r'M$_0$ (%s km/s)'%unit_dict[args.unit])

def plot_leaves():
    leafc = 'red'
    fig.show_contour(mask_hdu, colors=leafc, linewidths=1.0)    
    plt.plot([],[],color=leafc,label='Leaves')

def plot_branches():
    if num_conts > 1: 
        branchc = ['navy'] + ['lightblue'] * (num_conts - 1)
        branchls = [':'] + ['-'] * (num_conts - 1)
    else:
        branchc = ['lightblue']
        branchls = ['-']
    print (branchc)
    for i in range(num_conts): fig.show_contour(mask_hdu_bran[i],
                                                colors=branchc[i],
                                                linestyles = branchls[i],
                                                linewidths=0.8)
    if num_conts > 0:
        plt.plot([],[], branchls[0], color=branchc[0],label='Branch: {bran.idx}'.format(bran = minlvl_branches[0]))
        if num_conts > 1:
            plt.plot([],[],color=branchc[1],
                     label='Branches: {}'.format( ('{bran[%d].idx}, '*(num_conts - 1)%tuple(np.arange(1,num_conts))).format(bran = minlvl_branches))[:-2] )

def plot_legend():
    plt.legend(loc=(-23.7,0.9), framealpha = 0.97, fontsize = 8.5, fancybox = True)

def plot_dendro():
    #plt.savefig("img_moment0_%s_%s.png"%(args.unit,args.incl))#, dpi = 500)
    plot_leaves(); plot_branches(); plot_legend()
    output = "img_moment0dendro_%s_%s.png"%(args.unit,args.incl)
    plt.savefig(output)#, dpi = 500)
    print ('Saving figure on', output)

fig.tick_labels.set_xformat('dd')
fig.tick_labels.set_yformat('dd')

plot_dendro()
#plt.show()
