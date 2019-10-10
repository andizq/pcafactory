import numpy as np
import matplotlib.pyplot as plt
from turbustat.statistics import PCA
from astropy.io import fits
import astropy.units as u
from argparse import ArgumentParser
import sys
from utils import get_rt_mode

#**************************
#USER's INPUT
#**************************
parser = ArgumentParser(prog='PCA statistics', description='PCA statistics using TurbuStat')
parser.add_argument('-i', '--incl', default='faceon', help="Image inclination ['faceon', 'edgeon', 'edgeon_phi90'] to compute the moment from")
parser.add_argument('-f', '--folder', default='../../', help="Location of the original (whole) cube")
parser.add_argument('-u', '--unit', default='jypxl', help="Intensity units [jypxl, kelvin, tau]")
parser.add_argument('-n', '--portionid', default=0, type=int, help='Name id of the image portion.')
parser.add_argument('-c', '--cloud', default=0, type=int, help='Compute PCA from the entire region.')
#parser.add_argument('-o', '--output', help='Output name for the pca images + incl.; takes effect only if --name is set')

args = parser.parse_args()
if args.folder[-1] != '/': args.folder += '/'
#**************************
#TAGGING INPUT AND OUTPUT
#**************************
output = 'pca_turbustat_'+args.incl
if args.cloud: cubename = args.folder+"img_CO_J1-0_%s_%s_%s.fits"%(get_rt_mode(), args.unit, args.incl)
else: cubename = "./img_%s_%s_portion_%03d.fits"%(args.unit, args.incl, args.portionid)

try:
    pars = np.loadtxt('pars_pca.txt', dtype=np.str, delimiter=None, unpack=True) 
    if args.unit == 'jypxl': id = pars[0] == args.incl 
    else: id = pars[0] == args.incl+args.unit
    if args.cloud: col = 2
    else: col = 1
    min_eigval = float(pars[col][id])
except OSError:
    min_eigval = 0.99
print ("eigen_cut_method = 'proportion', min_eigval =", min_eigval)
#**************************
#PCA ANALYSIS
#**************************
cube = fits.open(cubename)[0]
if args.cloud: cube.header['CUNIT3'] = 'm/s'
source_dist = float(np.loadtxt(args.folder+'pars_size_rt.txt')[1])
pc2au = 206264.806247

pca = PCA(cube, distance = source_dist*u.pc)


print ('------------------------------ finished PCA from cube ------------------------------')
#5e1, seems that 2e2 is no longer needed
pca.run(verbose=True, min_eigval=min_eigval, spatial_output_unit=u.pc, spectral_output_unit=u.km/u.s, 
        brunt_beamcorrect=False, fit_method='odr', save_name=output+'_odr.png'
        , eigen_cut_method='proportion'
        )
#pca.fit_plaw(fit_method='odr',verbose=True)
print ('Number of eigenvalues:', pca.n_eigs)
print ('------------------------------ finished pca.run - method odr ------------------------------')

#**************************
#PCA ANALYSIS
#**************************

spatial_tmp = pca.spatial_width(unit=u.pc).value 
spectral_tmp = pca.spectral_width(unit=u.km/u.s).value
spat_error_tmp = pca.spatial_width_error(unit=u.pc).value 
spec_error_tmp = pca.spectral_width_error(unit=u.km/u.s).value

are_finite = (np.isfinite(spectral_tmp) * 
              np.isfinite(spatial_tmp) *   
              np.isfinite(spat_error_tmp) * 
              np.isfinite(spec_error_tmp) )

col_finite = lambda A: A[are_finite]

spatial = np.log10(spatial_tmp) # * abs(cube.header['CDELT1']) * 3600 * dpc / pc2au)
spectral = np.log10(spectral_tmp) # * cube.header['CDELT3'] * 1e-3)
spatial_error = 0.434 * spat_error_tmp / spatial_tmp
spectral_error = 0.434 * spec_error_tmp / spectral_tmp

#ind = np.logical_and(~np.isnan(spatial),~np.isnan(spectral))

columns = col_finite(np.array([spatial, spectral, spatial_error, spectral_error]).T)

if args.cloud: output_points = 'pca_%s_%s_cloud.txt'%(args.unit, args.incl)
else: output_points = 'pca_%s_%s_portion_%03d.txt'%(args.unit, args.incl, args.portionid)
np.savetxt(output_points, columns)
