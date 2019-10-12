import numpy as np
from regions import read_ds9, write_ds9
from astropy.io import fits
import astropy.units as u
from argparse import ArgumentParser
from utils import get_rt_mode

#**************************
#USER's INPUT
#**************************
parser = ArgumentParser(prog='Slicing a data cube', description='Slicing an image based on a ds9 box-region input')
parser.add_argument('-i', '--incl', default='faceon', help="Image inclination ['faceon', 'edgeon', 'edgeon_phi90'] to compute the moment from")
parser.add_argument('-f', '--folder', default='../', help="Location of the input data cube")
parser.add_argument('-u', '--unit', default='jypxl', help="Intensity units [jypxl, kelvin, tau]")
parser.add_argument('-r', '--region', help='Region file to be loaded.')
parser.add_argument('-R', '--rtmode', default='nonLTE', help="Radiative transfer mode [LTE, nonLTE]")

args = parser.parse_args()
if args.folder[-1] != '/': args.folder += '/'
#**************************
#TAGGING INPUT AND OUTPUT
#**************************
folder_reg = './portions_moment0/' #output folder

#**************************
#READING AND WRITING FILES
#**************************
cube = fits.open(args.folder+'img_CO_J1-0_%s_%s_%s.fits'%(args.rtmode, args.unit, args.incl))[0]
cube.header['CUNIT3'] = 'm/s'

regionname = 'clouds_boxes_%s_%s.reg'%(args.unit, args.incl)
if args.region is not None: regionname = str(args.region)
else: regionfile = open(folder_reg+regionname, 'r')

txt = regionfile.readlines()

for i,line in enumerate(txt[2:]):
    cx,cy,dx,dy,angle = np.array(line.splitlines()[-1].split('box')[-1][1:-1].split(',')).astype('float')
    limx_m = int(round(cx-0.5*dx))
    limx_p = int(round(cx+0.5*dx))
    limy_m = int(round(cy-0.5*dy))
    limy_p = int(round(cy+0.5*dy))
    output = 'img_%s_%s_portion_%03d.fits'%(args.unit,args.incl,i)
    newdata = cube.data[:,:,limy_m:limy_p, limx_m:limx_p]
    fits.writeto(folder_reg+output, newdata, cube.header, overwrite=True)
