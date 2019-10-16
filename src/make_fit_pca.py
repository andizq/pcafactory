from __future__ import print_function
import numpy as np
from scipy.optimize import curve_fit
from astropy.io import fits
import astropy.units as u
import matplotlib
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
from argparse import ArgumentParser
from functools import reduce
import os

def get_cloud_history(add=None, arepo_step=0.2):
    #arepo_step in Myr
    cwd = os.getcwd()
    cloud = cwd.split('cld')[1][0]

    rt_dict = {'Lime-arepo': 'LTE',
               'Polaris-arepo': 'LVG',
               'Lime-arepo-nonLTE': 'nonLTE'}

    rt_mode = rt_dict[cwd.split('RT-Analysis/')[1].split('/')[0]]
    
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
    return r'%s, complex %s, %.1f Myr old $-$ %s'%(case, cloud, time, rt_mode) + extras

def write_pca_data(data, kind, colors, naccepted, nrejected, file='data_pca.txt'):
    reduce_append = lambda x,y: np.append(x,y,axis=1) 
    kind = np.asarray(kind)
    colors = np.asarray(colors)
    data_all = reduce(reduce_append, [data, kind[None].T, colors[None].T])
    header = 'portions: accepted %d, rejected %d\nlog10_spatial-scale log10_velocity-scale spatial-error velocity-error kind color'%(naccepted, nrejected)
    np.savetxt(file, data_all, header=header, fmt='%s', delimiter='\t')

def write_fits_data(data, runs, points, kind, colors, naccepted, nrejected, file='data_fits.txt'):
    reduce_append = lambda x,y: np.append(x,y,axis=1) 
    runs = np.asarray(runs, dtype=np.str)
    points = np.asarray(points, dtype=np.str)
    kind = np.asarray(kind)
    colors = np.asarray(colors)
    data_all = reduce(reduce_append, [data, runs[None].T, points[None].T, kind[None].T, colors[None].T])
    header = 'portions: accepted %d, rejected %d\ncartesian-scaling-coefficient scaling-exponent std-coefficient std-exponent fit-runs fit-npoints kind color'%(naccepted, nrejected)
    np.savetxt(file, data_all, header=header, fmt='%s', delimiter='\t')

def func(x,b,m):
    return b+m*x

def func_heyer(x):
    return np.log10(0.87)+0.65*x

def make_fit(x,y,x_err,y_err,n=100):    
    popt, pcov = [],[]
    for i in range(n):
        x_g = np.random.normal(loc=x, scale=x_err)
        y_g = np.random.normal(loc=y, scale=y_err)
        pt,pv = curve_fit(func, x_g, y_g)
        popt.append(pt)
        pcov.append(pv)
    return popt, pcov
 
lnof10 = np.log(10)
def coeff_log2cart(log_coeff, log_errors):
    coeff = np.asarray(log_coeff)
    errors = np.asarray(log_errors)
    cart_coeff = np.power(10,coeff)
    cart_errors = [None]*len(log_errors)
    if errors != np.asarray(cart_errors): cart_errors = cart_coeff*errors*lnof10
    return cart_coeff, cart_errors
  
def convert_ticks(ax): #log labels to cart labels
    xlims = np.array(ax.get_xlim())
    ylims = np.array(ax.get_ylim())
    minors = np.log10( reduce(np.append, [np.arange(i*0.2, i*1.0, i*0.1) for i in [0.1, 1,10,100]] ))
    minor_xlabels = np.log10(np.array([0.3,0.5,3.0,20.0]))
    minor_ylabels = np.log10(np.array([]))
    majors = np.log10([0.1,1.0,10.0])
    major_xlabels = majors
    major_ylabels = majors
    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_minor_locator(ticker.FixedLocator(minors))    
        axis.set_major_locator(ticker.FixedLocator(majors))

    def fmt_func(x,pos, minor_labels, major_labels, lims):
        if ((1e-10 > abs(x-minor_labels)).any() or (1e-10 > abs(x-major_labels)).any()
             and (x>=lims[0] and x <=lims[1])): return '%.1f'%(10**x)
        else: return ''

    fmt_xaxis = ticker.FuncFormatter(lambda x,pos: fmt_func(x,pos,minor_xlabels,major_xlabels,xlims))    
    fmt_yaxis = ticker.FuncFormatter(lambda x,pos: fmt_func(x,pos,minor_ylabels,major_ylabels,ylims))    

    ax.xaxis.set_minor_formatter(fmt_xaxis)
    ax.xaxis.set_major_formatter(fmt_xaxis)
    ax.yaxis.set_minor_formatter(fmt_yaxis)
    ax.yaxis.set_major_formatter(fmt_yaxis)
   
def add_label(ax, xy, text, **kwargs):
    kwargs_def = {'color': 'k', 'size': MEDIUM_SIZE-1, 'ha': 'center', 'family': 'sans-serif'}
    kwargs_def.update(kwargs)
    y = xy[1] - 0.15  # shift y-value for label so that it's below the artist
    ax.text(xy[0], y, text, **kwargs_def)
  
def plot_ax0(ax0,img_moment0):
    ax0.imshow(img_moment0)
    ax0.axis('off')

def plot_ax1_heyer(ax1,x):
    ax1.plot(x, func_heyer(x), 'r-', lw=2.5, label = 'Heyer+2004')
    ax1.set_xlabel(r'spatial scale [pc]', fontsize=MEDIUM_SIZE-1, labelpad=6)
    ax1.set_ylabel(r'velocity scale [km/s]', fontsize=MEDIUM_SIZE-1)

def plot_ax1_scatter(ax1,x,y,x_err=None,y_err=None, marker='o', s=50, colors='white', ecolors='black', alpha_errbar=0.25):
    ax1.scatter(x, y, facecolors = colors, marker=marker, s=s, edgecolors = 'black', alpha=1.0, zorder=3)
    ax1.errorbar(x, y, xerr=x_err, yerr=y_err, color='None', ecolor = ecolors, linestyle='none', elinewidth=2, alpha=alpha_errbar)

def plot_ax1_fit(ax1,data_sim,popt_list,pcov_list, ls='--', lw=2.5, alpha=1.0, fill_alpha=0.2, fill_lw=1.5, color='blue', label='', show_eq=False, convert_ticks=True):    
    popt, pcov = np.asarray(popt_list), np.asarray(pcov_list)
    popt_m = np.mean(popt_list, axis=0)
    popt_std = np.std(popt, axis=0)

    curve_mean = func(data_sim, *popt_m)
    
    ncurves = func(data_sim[np.newaxis].T, popt[:,0], popt[:,1])
    curve_std = np.std(ncurves, axis=1)
    curve_ind_min, curve_ind_max = curve_mean - curve_std, curve_mean + curve_std

    if show_eq: 
        [cart_x], [cart_errors] = coeff_log2cart([popt_m[0]], [popt_std[0]])
        add_label(ax1, (0.5, 1.06), r"Mixed fit", size=SMALL_SIZE+2, family='monospace', transform=ax1.transAxes)
        add_label(ax1, (0.5, 1.0), r"$\delta\upsilon = (%.2f\pm%.2f)\mathscr{l}^{(%.2f\pm%.2f)}$"%(cart_x,cart_errors, popt_m[1],popt_std[1]), 
                  size=SMALL_SIZE+2, transform=ax1.transAxes)#, bbox=dict(facecolor='white', alpha=0.5))

    ax1.tick_params(which='both', labelsize = SMALL_SIZE+2)
    
    line1, = ax1.plot(data_sim, curve_mean, linestyle=ls, color=color, linewidth=lw, alpha=alpha, label=label)
    ax1.fill_between(data_sim, curve_ind_min, curve_ind_max, edgecolor=color, facecolor=color, linewidth=fill_lw, alpha=fill_alpha)
    
    return popt_m, popt_std
    
def plot_ax2_rejected(ax2):
    ax2.set_title('Points per\nrejected-cloud', pad=5, fontsize=SMALL_SIZE+1)
    ax2.set_xlim(-0.1,2.1); ax2.set_ylim(-0.1,1.1)
    x_coord = np.linspace(0.0,2.0,num=nrejected+1)
    widths = x_coord[1:] - x_coord[:-1]
    patches = []

    for i in range(nrejected):
        x_patch = x_coord[i]
        patches.append(mpatches.FancyBboxPatch(
                (x_patch,0.0), widths[i], 1.0, 
                boxstyle=mpatches.BoxStyle("Round", pad=0.0, rounding_size=0.2)))
        add_label(ax2, (x_patch + 0.5*widths[i],0.5), data_size[~ids_bool][i], color='white')
    if nrejected == 0: 
        patches.append(mpatches.FancyBboxPatch(
                (0.0,0.0), 2.0, 1.0, 
                boxstyle=mpatches.BoxStyle("Round", pad=0.0, rounding_size=0.2)))
        add_label(ax2, (1.0,0.5), 'None', color='black')

    collection = PatchCollection(patches, alpha=1.0)
    collection.set_edgecolor('k')
    collection.set_facecolor(colors_all[~ids_bool])
    ax2.add_collection(collection)
    ax2.axis('off')

def plot_ax3_pie(ax3, ndata):
    # Pie chart, where the slices will be ordered and plotted counter-clockwise:
    ax3.set_title('Points per cloud', pad=5, fontsize=SMALL_SIZE+1)
    labels = data_size[ids_bool] 
    fracs = data_size[ids_bool]
    #explode = (0, 0, 0, 0)  # Don't "explode" any slice
    patches,texts = ax3.pie(fracs, labels=labels, autopct=None, labeldistance=0.75, 
                            rotatelabels=True*0, colors = colors_accepted, center = (0.0,0.0),
                            shadow=False, startangle=90, frame=True*0, radius=1, wedgeprops={'width': 0.6, 'edgecolor': 'k'})
    plt.setp(texts, fontsize=TINY_SIZE+1, color='white', fontweight='bold')
    #ax3.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    add_label(ax3, (0.0,0.0), ndata)
    ax3.set_xlim(-1,1); ax3.set_ylim(-1,1)

def plot_ax4_pca_scatter(ax4, x, y, x_err=None, y_err=None, do_log2cart=True, fc='black', marker='P', label='', s=50, **kwargs):
    if do_log2cart: [cart_x], [cart_errors] = coeff_log2cart([x], [x_err])
    else: cart_x, cart_errors = x, x_err
    ax4.scatter(cart_x, y, facecolors = fc, marker = marker, s=s, edgecolors = 'black', alpha=1.0, label=label, **kwargs)
    
    if np.array_equal(fc,'white') or np.array_equal(fc, [1,1,1,1]): fc = 'black'; alpha_errorbar = 0.7
    else: alpha_errorbar = 0.25
    ax4.errorbar(cart_x, y, xerr=cart_errors, yerr=y_err, color='None', ecolor = fc, 
                linestyle='none', elinewidth=2, alpha=alpha_errorbar, zorder=-1)
    return cart_x, y, cart_errors, y_err

def plot_ax4_mean(ax4, fit_pars_portions, fc='none', marker='o', label='clds mean', s=90, ls='--', lw=1, **kwargs): 
    #I wanted to draw the mean of the portions as a tiny pie, however it seems to over-saturate the image with info.
    fit_pars = np.asarray(fit_pars_portions)
    center = np.average(fit_pars[:,0:2], axis=0, weights=data_size[ids_bool])
    std_dv = np.sqrt(np.average((fit_pars[:,0] - center[0])**2, weights=data_size[ids_bool]))
    std_alpha = np.sqrt(np.average((fit_pars[:,1] - center[1])**2, weights=data_size[ids_bool]))
    
    ax4.scatter(*center, facecolors=fc, marker=marker, s=s, edgecolors = 'black', alpha=1.0, label=label, linestyles=ls, linewidths=lw, **kwargs)

    add_label(ax4, (0.25,0.199), r'$\sigma_{\upsilon_0}=%.2f$, $\sigma_\alpha=%.2f$ from '%(std_dv, std_alpha), fontsize=SMALL_SIZE, transform=ax4.transAxes)
    ax4.scatter(0.5, 0.2105-0.15, facecolors=fc, marker=marker, s=s, edgecolors = 'black', alpha=1.0, 
                linestyles=ls, linewidths=lw, transform=ax4.transAxes, **kwargs)

    patch = mpatches.FancyBboxPatch((0.01,0.01), 0.53, 0.1, facecolor='none', linewidth=1.4, 
                                    transform=ax4.transAxes, boxstyle=mpatches.BoxStyle("Round", pad=0.0, rounding_size=0.03))
    ax4.add_artist(patch)


def plot_ax4_mean_pie(ax4, fit_pars_portions, max_lim):
    fit_pars = np.asarray(fit_pars_portions)
    pie_center = np.mean(fit_pars[:,0:2], axis=0)
    pie_std = np.std(fit_pars[:,0:2], axis=0)
    fracs = data_size[ids_bool]
    patches,texts = ax4.pie(fracs, labels=None, autopct=None, 
                            rotatelabels=False, colors = colors_accepted, center = pie_center,
                            shadow=False, startangle=90, frame=True, radius=max_lim*0.018, wedgeprops={'edgecolor': 'k'})
    plt.setp(texts, fontsize=TINY_SIZE+1, color='white', fontweight='bold')
    #ax4.errorbar(*pie_center, *pie_std, color='None', ecolor='k',
    #              linestyle='none', elinewidth=2, alpha=0.7, zorder=-1)
    #ax4.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

if __name__ == '__main__':

    matplotlib.rcParams['font.family'] = 'monospace'
    matplotlib.rcParams['axes.linewidth'] = 1.8  
    matplotlib.rcParams['xtick.major.width']=1.5
    matplotlib.rcParams['ytick.major.width']=1.5
    matplotlib.rcParams['xtick.minor.width']=1.
    matplotlib.rcParams['ytick.minor.width']=1.
    matplotlib.rcParams['xtick.major.size']=5.0
    matplotlib.rcParams['ytick.major.size']=5.0
    matplotlib.rcParams['xtick.minor.size']=2.8
    matplotlib.rcParams['ytick.minor.size']=2.8

    TINY_SIZE = 8
    SMALL_SIZE = 10
    MEDIUM_SIZE = 15
    BIGGER_SIZE = 20

    #**************************
    #USER's INPUT
    #**************************
    parser = ArgumentParser(prog='Fit PCA', description='Fit to PCA points from TurbuStat')
    parser.add_argument('-i', '--incl', default='faceon', help="Image inclination ['faceon', 'edgeon', 'edgeon_phi90'] to compute the moment from")
    parser.add_argument('-u', '--unit', default='jypxl', help="Intensity units [jypxl, kelvin, tau]")

    args = parser.parse_args()

    tag_tuple = (args.unit, args.incl)
    portions = os.popen('ls *img*%s_%s_portion*'%tag_tuple).readlines()
    nportions = len(portions)

    portion_files = os.popen('ls *pca_%s_%s_portion*'%tag_tuple).readlines()
    nfiles = len(portion_files)

    img_moment0 = plt.imread('../img_moment0_%s_%s.png'%tag_tuple)

    reduce_append = lambda x,y: np.append(x,y,axis=0)

    data,ids = [],[]
    data_size = np.zeros(nportions, dtype=np.int)
    ids_bool = np.zeros(nportions, dtype=np.bool)
    file_overlaped = np.loadtxt('overlaped_portions.txt', dtype=np.str, delimiter=None)
    if args.unit=='jypxl': overlaped_portions = np.array(file_overlaped[:,1][file_overlaped[:,0] == args.incl][0].split(',')).astype(np.int32)
    else: overlaped_portions = np.array(file_overlaped[:,1][file_overlaped[:,0] == args.incl+args.unit][0].split(',')).astype(np.int32)

    for file in portion_files:
        name = file.split('\n')[0]
        i = int(name[-7:-4]) #portion id
        if i in overlaped_portions:
            data_size[i] = len(np.loadtxt(name))
            continue
        data.append(np.loadtxt(name))
        data_size[i] = len(data[-1])
        ids.append(i)
        ids_bool[i] = True
    
    naccepted = len(data_size[ids_bool]) #Accepted portions
    nrejected = len(data_size[~ids_bool])

    colors_all = np.loadtxt('clouds_colors_%s_%s.txt'%tag_tuple, dtype=float)
    colors_accepted = colors_all[ids]
    colors = [np.array([colors_all[id]]*data_size[id]) for id in ids]
    colors = reduce(reduce_append,colors)
    colors_str = [np.str(c) for c in colors]
    
    nruns_mixed = 1000
    nruns_cloud = 1000
    nruns_portions = 200
    
    data_portions = data #Portions
    data = reduce(reduce_append,data) #Mixed portions
    ndata = len(data)
    x,y,x_err,y_err = data.T
    popt, pcov = make_fit(x,y,x_err,y_err,n=nruns_mixed) #Mixed fit
    
    data_cloud = np.loadtxt('pca_%s_%s_cloud.txt'%tag_tuple)
    ndata_cloud = len(data_cloud)
    color_cloud = [1.,1.,1.,1.]
    x_c,y_c,x_err_c,y_err_c = data_cloud.T
    popt_c, pcov_c = make_fit(x_c,y_c,x_err_c,y_err_c,n=nruns_cloud) #Cloud fit

    data_all = np.append(data, data_cloud, axis=0)
    kind = reduce(lambda x,y: x+y, [['portion%d'%id]*data_size[id] for id in ids])
    kind += ['cloud']*ndata_cloud
    write_pca_data(data_all, kind, colors_str + [np.str(color_cloud)]*ndata_cloud, naccepted, nrejected, file='data_pca_%s_%s.txt'%tag_tuple)
    
    data_sim = np.linspace(np.min(x)-0.15, np.max(x)+0.15, 100) #Synthetic x for Heyer and fit equations.

    #********
    #PLOTTING
    #********
    size_inches = (12.5,5.0)
    fig = plt.figure(figsize = size_inches)    
    ax0 = fig.add_axes([0.58,0.01,0.38,(0.38)*size_inches[0]/size_inches[1]])           #Moment 0
    ax1 = fig.add_axes([0.07,0.1,0.5,0.8])                                              #Fit plots
    ax2 = fig.add_axes([0.275, 0.11, 0.12, 0.05*reduce(lambda x,y: x/y, size_inches)]) #Rejected portions
    ax3 = fig.add_axes([0.095, 0.59, 0.1, 0.1*reduce(lambda x,y: x/y, size_inches)])    #Accepted portions (pie chart)

    fig2 = plt.figure(figsize = (5.0,5.0))
    ax4 = fig2.add_axes([0.1,0.1,0.8,0.8]) #alpha vs coeff

    plot_ax0(ax0,img_moment0)
    
    fit_pars_portions, fit_pars_heyer, fit_pars_mixed, fit_pars_cloud = [], [], [], []
    coeff_portions, index_portions = [], []
    for i in range(len(data_portions)):
        xp,yp,xp_err,yp_err = data_portions[i].T
        poptp, pcovp = make_fit(xp,yp,xp_err,yp_err,n=nruns_portions) 
        poptp_m, poptp_std = plot_ax1_fit(ax1, data_sim, poptp, pcovp, 
                                          ls='-', lw=0.0, alpha=0.0, fill_alpha=0.04, fill_lw=0.0, color='black'
                                          )    
        fit_pars_portions.append(plot_ax4_pca_scatter(ax4,*poptp_m, *poptp_std, marker='o', fc=colors_accepted[i], label='cld %d'%ids[i], zorder=3))
        coeff_portions.append([poptp_m[0], poptp_std[0]]) #log10 coefficients from portions
        index_portions.append([poptp_m[1], poptp_std[1]]) #indices from portions
    coeff_portions = np.asarray(coeff_portions)
    index_portions = np.asarray(index_portions)

    fit_heyer = func_heyer(data_sim)
    #ax1.set_title(get_cloud_history(add=[args.incl]), fontsize=SMALL_SIZE)
    ax1.set_xlim(np.min(data_sim), np.max(data_sim))
    ax1.set_ylim(np.min(fit_heyer)-1.0, np.max(fit_heyer)+0.9)
    plot_ax1_heyer(ax1,data_sim)
    popt_m, popt_std = plot_ax1_fit(ax1,data_sim,popt,pcov, show_eq=True, label='Mixed fit', ls='--', lw=2.2)
    poptc_m, poptc_std = plot_ax1_fit(ax1,data_sim,popt_c,pcov_c, show_eq=False, ls='--', lw=1.5, 
                                      color='darkgreen', alpha=1.0, fill_alpha=0.07, label='Complex fit')
    plot_ax1_scatter(ax1,x,y,x_err,y_err, colors=colors, ecolors=colors, alpha_errbar=0.28-0.08*(ndata/100.))
    plot_ax1_scatter(ax1,x_c,y_c,x_err_c,y_err_c, colors=color_cloud, s=80, marker='X', alpha_errbar=0.35-0.08*(ndata_cloud/20.))
    ax1.scatter(None,None, marker='o', facecolors='white', edgecolors='black', s=50, label='PCA-clouds')
    ax1.scatter(None,None, marker='X', facecolors=color_cloud, edgecolors='black', s=80, label='PCA-complex(%2d)'%ndata_cloud)
    ax1.legend(loc='lower right', fontsize = MEDIUM_SIZE-3, borderpad=None, labelspacing=0.3, handlelength=1.7)    
    convert_ticks(ax1)
    plot_ax2_rejected(ax2)
    plot_ax3_pie(ax3, ndata)
    plot_ax4_mean(ax4, fit_pars_portions, zorder=3) 

    fit_pars_heyer.append(plot_ax4_pca_scatter(ax4,0.87, 0.65, 0.01, 0.01, s=150, marker=r'$\mathcircled{H}$', lw=0, fc='black', label='Heyer+2004', do_log2cart=False, zorder=6))
    fit_pars_mixed.append(plot_ax4_pca_scatter(ax4,*popt_m,*popt_std, s=100, marker='P', fc='white', label='Mixed fit', zorder=6))
    fit_pars_cloud.append(plot_ax4_pca_scatter(ax4,*poptc_m,*poptc_std, s=100, marker='X', fc=color_cloud, label='Cmplx fit', zorder=6))

    fits_data = reduce(reduce_append, [fit_pars_heyer, fit_pars_mixed, fit_pars_cloud, fit_pars_portions])
    fits_runs = [1, nruns_mixed, nruns_cloud] + [nruns_portions]*naccepted
    fits_points = [0, np.sum(data_size[ids_bool]), ndata_cloud] + list(data_size[ids_bool])
    fits_kind = ['heyer+2004', 'mixed', 'cloud'] + ['portion%d'%id for id in ids]
    fits_colors = ['red', 'blue', 'darkgreen'] + [np.str(c) for c in colors_accepted]
    write_fits_data(fits_data, fits_runs, fits_points, fits_kind, fits_colors, naccepted, nrejected, file='data_fits_%s_%s.txt'%tag_tuple)

    #ax4.set_title(ax1.get_title(), fontsize=TINY_SIZE-1.5)
    ax4_max_lim = np.max(fits_data[:,0:2], axis=0) + 0.2
    #plot_ax4_mean_pie(ax4, fit_pars_portions, ax4_max_lim)
    ax4.set_xlim(0, ax4_max_lim[0])
    ax4.set_ylim(0, ax4_max_lim[1])
    ax4.set_xlabel(r'$\upsilon_0$ [km/s]', fontsize=BIGGER_SIZE, labelpad=5)
    ax4.set_ylabel(r'$\alpha_{\rm PCA}$', fontsize=BIGGER_SIZE, labelpad=5)
    #ax4.legend(framealpha=0.5, fontsize=TINY_SIZE+1, loc='best', bbox_to_anchor=(0., 0.15, 1.0, 0.85)) #Choose the best loc for ax4.legend above y=0.15
    ax4.legend(framealpha=0.5, fontsize=TINY_SIZE+1, loc='lower left', bbox_transform=ax4.transAxes, bbox_to_anchor=(1.005, -0.019))
    output = 'img_fit_%s_%s_allportions.png'%tag_tuple
    output2 = 'img_pars_%s_%s_offsets.png'%tag_tuple
    fig.savefig(output, dpi=500, bbox_inches='tight')
    fig2.savefig(output2, dpi=500, bbox_inches='tight')
    print ('Saving pca fit on',output)
    print ('Saving pca pars on',output2)
    #plt.show()
