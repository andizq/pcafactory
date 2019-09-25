import numpy as np
from spectral_cube import SpectralCube
from argparse import ArgumentParser
from astropy.wcs import WCS
from astropy.io import fits
from utils import get_rt_mode

parser = ArgumentParser(prog='Computing moment 0', description='Compute moment 0 from data cube')
parser.add_argument('-i', '--incl', default='faceon', help="Image inclination ['faceon', 'edgeon', 'edgeon_phi90'] to compute the moment from")
parser.add_argument('-f', '--folder', default='../', help="Location of the input data cube")
parser.add_argument('-u', '--unit', default='jypxl', help="Intensity units [jypxl, kelvin, tau]")

args = parser.parse_args()
if args.folder[-1] != '/': args.folder += '/'

img_name = 'img_CO_J1-0_%s_%s_%s.fits'%(get_rt_mode(), args.unit, args.incl)
fn = args.folder + img_name
out_name = 'moment0_' + img_name

hdu = fits.open(fn)[0]
hdu.header['CUNIT3'] = 'm/s'
w = WCS(hdu.header)

cube = SpectralCube(data=hdu.data.squeeze(), wcs=w.dropaxis(3))
#cube = SpectralCube.read(fn) #Fails due to the 'm/S' in the header. It needs to be first corrected and then create manually the SpectralCube instance

m0 = cube.moment(order=0)
m0.write(out_name, overwrite=True)
print ('output:', out_name)
