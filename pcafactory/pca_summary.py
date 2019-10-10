import numpy as np
import json
import os
import sys

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

markers = {'faceon': 's',
           'edgeon': 'D',
           'edgeon_phi90': 'o',
           'edgeon_phi90tau': 'D'}

size_0 = {'s': 100,
          'D': 90,
          'o': 120}

scenarios = {'pot230nSG': 'potential-only',
             'pot230SGjeans': 'self-gravity',
             'Fjeans248': 'SN-feedback'}

colors_dict = {'potential-only': '#648FFF',
               'self-gravity': '#DC267F',
               'SN-feedback': '#FFB000'}

kind_dict = {'mixed': 'Mixed clouds\nfrom complexes',
             'cloud': 'Whole complexes,\nno portioning',
             'portions': 'Individual clouds\nfrom complexes'}

root = '/Users/andizq/Downloads/Manchester_2018/Rowan-data/Andres/data/RT-Analysis/'
path = root+'%s/%s_cld%s-Lime/extracted_%s/Subgrids/Dendrograms_portions/portions_moment0/data_fits_%s_%s.txt'


def size_markers(size_array, marker='s'):
    return size_0[marker]*size_array**0.8

def get_cloud_history(path, add=None, arepo_step=0.2, ret_label=False):
    #arepo_step in Myr
    cloud = path.split('cld')[1][0]

    rt_dict = {'Lime-arepo': 'LTE',
               'Polaris-arepo': 'LVG',
               'Lime-arepo-nonLTE': 'nonLTE'}

    rt_mode = rt_dict[path.split('RT-Analysis/')[1].split('/')[0]]
    
    case_dict = {'Fjeans': 'Self-Gravity + Mixed Supernovae', 
                 'nSG': 'No Self-Gravity + Random Supernovae', 
                 'SGjeans': 'Self-Gravity + Random Supernovae'}
    for key in case_dict: 
        if key in path: case = case_dict[key]; break

    if key == 'Fjeans': pot_time = int(path.split(key)[1][0:3])
    else: pot_time = int(path.split(key)[0][-3:])
    cloud_time = int(path.split('extracted_')[1][0:3])
    time = (cloud_time - pot_time) * arepo_step
    extras = ''
    if add != None:
        for a in add: extras+= ' $-$ %s'%a

    if ret_label: return r'%s, cloud %s, %.1f Myr old $-$ %s'%(case, cloud, time, rt_mode) + extras
    else: return case, cloud, time, rt_mode

def plot_literature(ax, authors2plot='all', s=250, fontsize_patch=10):    
    xlims = ax.get_xlim()
    for entry in literature_dict:
        if entry not in authors2plot and authors2plot!='all': continue
        pars = (literature_dict[entry]['v0'], literature_dict[entry]['gamma'])
        errors = (literature_dict[entry]['dv0'], literature_dict[entry]['dgamma']) 
        kwargs = literature_dict[entry]['kwargs']
        kind = literature_dict[entry]['kind']
        if kind == 'point':
            ax.scatter(*pars, s=s, linewidths=0.0, facecolor='k', edgecolor='k', zorder=3, label=entry, alpha=1.0, **kwargs)
            ax.errorbar(*pars, *errors[::-1], ecolor = 'k', linestyle='none', elinewidth=3, alpha=0.3, zorder=0)                
        if kind == 'line':
            ax.axhline(pars[1], lw=2 , label=entry, **kwargs)
            if errors[1] is not None: ax.fill_between(xlims, pars[1]+errors[1], pars[1]-errors[1], edgecolor='none', facecolor=kwargs['color'], lw=0, alpha=0.1)
        if kind == 'patch':
            marker_label = literature_dict[entry]['marker_label']
            #There is no an easy way to combine patch+text in a single object to then show it in the plot legend. If want to show legend of patches 
            # I should rather make a new axes instance and draw the markers+text manually.  
            #patch = mpatches.Circle(None,None, radius=0.1, linewidth=1, facecolor='none', edgecolor='k', label=entry, alpha=1.0, **kwargs)
            #ax.add_artist(patch)
            ax.scatter(*pars, s=s, linewidths=1.6, facecolor='none', edgecolor='k', zorder=3, label=entry, alpha=1.0, **kwargs)
            ax.text(*pars, marker_label, fontsize = fontsize_patch, weight='bold', horizontalalignment='center', verticalalignment='center')            
            ax.errorbar(*pars, *errors[::-1], ecolor = 'k', linestyle='none', elinewidth=3, alpha=0.3, zorder=0)                
            
def plot_sizeaxes(ax, tick_params = dict(direction='in', left=False, labelleft=False, top=True)):
    ax.set_xlim(0.5,4.5)
    ax.set_ylim(-1.2,1.2)
    ax.set_xlabel('Snapshot time (Myr)', fontsize=SMALL_SIZE+2)
    ax.tick_params(**tick_params)
    x = np.arange(0,5)
    yu = np.zeros(len(x))+0.7
    ym = np.zeros(len(x))
    yd = np.zeros(len(x))-0.7
    mark_face = markers['faceon']
    mark_edge = markers['edgeon']
    mark_edge90 = markers['edgeon_phi90']
    ax.scatter(x,yu, marker=mark_face, edgecolor='k', facecolor='none', s=size_markers(x, mark_face), label='faceon')
    ax.scatter(x,ym, marker=mark_edge, edgecolor='k', facecolor='none', s=size_markers(x, mark_edge), label='edgeon')
    ax.scatter(x,yd, marker=mark_edge90, edgecolor='k', facecolor='none', s=size_markers(x, mark_edge90), label='edgeon_phi90')
    ax.text(4.7, 0.7-0.17, 'face-on', fontsize=MEDIUM_SIZE)
    ax.text(4.7, 0.0-0.17, r'edge-on$_{\phi=0^{\rm o}}$', fontsize=MEDIUM_SIZE)
    ax.text(4.7,-0.7-0.17, r'edge-on$_{\phi=90^{\rm o}}$', fontsize=MEDIUM_SIZE)

def plot_colour_legend(ax, fontsize=MEDIUM_SIZE, loc='lower right', 
                       labels = ('a. pot. dom','b. SG on', 'c. SN fdbck'), **kwargs):
    dum = [None]*2
    potential, = ax.plot(dum, dum, color=colors_dict['potential-only'], lw=10)
    selfgravity, = ax.plot(dum, dum, color=colors_dict['self-gravity'], lw=10)
    feedback, = ax.plot(dum, dum, color=colors_dict['SN-feedback'], lw=10)

    leg2 = ax.legend(handlelength=0.4, borderpad=0.8, handles=(potential, selfgravity, feedback), 
                     labels=labels, loc=loc, fontsize=fontsize, **kwargs)
    return leg2

def plot_cldcomplex_legend(ax,ax1): #To do: plot this in a different coord transf so that the pos remains constant under plot changes.
    newBracketB = mpatches.ArrowStyle.BracketB(widthB=0.7, lengthB=0.4)
    ax1.annotate("A",
                 xy=(1.3, -0.35), xycoords='axes fraction',
                 xytext=(0.856, 0.17), textcoords='axes fraction',
                 fontsize=SMALL_SIZE+2,
                 arrowprops=dict(arrowstyle=newBracketB, 
                                 color='0.2',
                                 linewidth=1.6, 
                                 shrinkA=5, shrinkB=6,
                                 patchA=None, patchB=None,
                                 connectionstyle='arc3,rad=0.6',
                                 ),
                 )
    ax.text(1.3+0.23,-0.35+0.1,'Cloud Complex', fontsize=SMALL_SIZE+2, ha='center', va='center', rotation=13, transform=ax1.transAxes) 
    #Drawn over ax but using ax1 coords, so it's independent on ax limits.

def plot_pars_histogram(ax, kind, data_dict, data_axis='gamma', pdf=False):
    axis_dict = {'v0': 0, 'gamma': 1}
    axis_or = {'v0': 'vertical', 'gamma': 'horizontal'}

    for case in data_dict: 
        data = data_dict[case][:,axis_dict[data_axis]]
        nbins = int(round(1+3.322*np.log10(len(data))))
        if kind != 'portions': nbins += -1
        if pdf: pass
        else: 
            n, bins, patches = ax.hist(data, bins=nbins, orientation=axis_or[data_axis], density=True, linewidth=1.5,
                                       facecolor=colors_dict[scenarios[case]], edgecolor='k', alpha=0.7)

def plot_pca_pars(ax, base, kind, print_path=False):
    i = 0
    data = []
    data_tag = {'cloud': [], 'data': [], 'time': [], 'orientation': []}
    case = base[i+1]
    for folder in base[i]:
        for cloud in base[i+2]:
            for time in base[i+3][::-1]: #Small markers over big ones
                for unit in base[i+4]:
                    for incl in base[i+5]:
                        path_base = path%(folder,case,cloud,time,unit,incl)
                        try: 
                            pars = np.loadtxt(path_base, dtype=np.str, delimiter='\t', unpack=False)
                            if kind=='portions': dv,alpha,dv_err,alpha_err = pars[3:,0:4].T.astype('float')
                            else: dv,alpha,dv_err,alpha_err = pars[:,0:4][pars[:,6]==kind].T.astype('float')                                
                            gamma, gamma_err = alpha_to_gamma(alpha, alpha_err)                            
                        except OSError: continue
                        if print_path: print (path_base)
                        
                        tmp = [dv,gamma,dv_err,gamma_err]
                        data.append(tmp)
                        time_Myr = get_cloud_history(path_base)[2]

                        data_tag['cloud'].append(cloud)
                        data_tag['data'].append(tmp)
                        data_tag['time'].append(time_Myr)
                        data_tag['orientation'].append(incl)

                        if unit != 'jypxl': incl = incl+unit
                        ax.scatter(dv,gamma, marker=markers[incl], facecolor=colors_dict[scenarios[case]], edgecolor='k', linewidths=1.1, s=size_markers(time_Myr, markers[incl]), alpha=0.8, zorder=3)
                        ax.errorbar(dv,gamma, xerr=dv_err, yerr=gamma_err, ecolor = colors_dict[scenarios[case]], linestyle='none', elinewidth=3, alpha=0.5, zorder=0)    
                        if kind!='portions': 
                            cloud_name = cloud
                            if case == 'pot230nSG': cloud_name += r'$_0$'
                            for xi,yi in zip(dv,gamma): ax.text(xi-0e-3,yi-4.5e-3, cloud_name, weight='normal',fontsize=SMALL_SIZE-0.5, horizontalalignment='center', verticalalignment='center')
    
    data = np.array(data).squeeze()
    if isinstance(data[0,0], np.ndarray): #i.e if more than 1 value per entry
        new_data = []
        for i in range(4): new_data.append(np.concatenate(data[:,i]).ravel())
        data = np.array(new_data).T
    return data, data_tag
               
             
def figure_main(kind, xlims=(0,1.7), ylims=(-0.5,1.5), ticks_freq=0.2, units=['jypxl']):
    fig, ax = plt.subplots(nrows=1, figsize=(7,7))
    ax.set_xlim(xlims)            
    ax.set_ylim(ylims)
    ax.set_xticks(np.arange(*xlims, ticks_freq))
    ax.set_yticks(np.arange(0,ylims[1], ticks_freq))
    ax.set_xlabel(r'$\upsilon_0$ [km/s]', fontsize=BIGGER_SIZE, labelpad=10)
    #font = matplotlib.font_manager.FontProperties(family='DejaVu Serif', style='italic', size=16)
    ax.set_ylabel(r'$\mathscr{\gamma}$', fontsize=BIGGER_SIZE, labelpad=10)#, fontproperties=font)

    incl = ['edgeon', 'faceon', 'edgeon_phi90']
    cloudsAB = ['A', 'B']
    cloudsCD = ['C', 'D']
    time_nSG = ['240', '250']
    time_SG = ['235', '240', '245']
    time_fdbck = ['253', '260']
    cases = ['pot230nSG', 'pot230SGjeans', 'Fjeans248']
    rt_folders = ['Lime-arepo-nonLTE']

    nSG_base = [rt_folders, cases[0], cloudsAB, time_nSG, units, incl]
    SG_base = [rt_folders, cases[1], cloudsAB, time_SG, units, incl]
    fdbck_base = [rt_folders, cases[2], cloudsCD, time_fdbck, units, incl]
    bases = [nSG_base, SG_base, fdbck_base]

    data_dict = {}
    data_tag = {}
    for case,base in zip(cases,bases): data_dict[case], data_tag[case] = plot_pca_pars(ax, base, kind)

    plot_literature(ax, authors2plot=['Kolmogorov+1941', 'Kraichnan+1974', 'Larson+1981', 'Solomon+1981','Heyer+2004', 'Roman-Duval+2011', 'Bertram+2014', 'Hacar+2016'])
    leg = ax.legend(handlelength=1.5, borderpad=0.5, ncol=2, fontsize=SMALL_SIZE+1, framealpha=1.0, loc='lower left', bbox_to_anchor=(-0.12, 0.006))    
    fig.gca().add_artist(leg)
    ax1 = fig.add_axes([0.15,0.87,0.4,0.2])
    plot_sizeaxes(ax1)
    leg2 = plot_colour_legend(ax)
    plot_cldcomplex_legend(ax,ax1)

    fig.savefig('PCA_summary_%s_%s.pdf'%(units[0],kind), dpi=500, bbox_inches='tight') 
    return data_dict, data_tag


def figure_histograms(kind, data_dict, xlims=(0,1.7), ylims=(-0.5,1.5), ticks_freq=0.2, units=['jypxl'], pdf=False, legend=False, separate_clds=False, log=False):
    fig = plt.figure(figsize=(9,9))
    ax = fig.add_axes([0.1,0.1,0.75,0.75])
    ax1 = fig.add_axes([0.85,0.1,0.15,0.75])
    ax2 = fig.add_axes([0.1,0.85,0.75,0.15])
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax1.set_ylim(ylims)
    ax2.set_xlim(xlims)
    ax1.axis('off')
    ax2.axis('off')
        
    ax.set_xticks(np.arange(*xlims, ticks_freq))
    ax.set_yticks(np.arange(0,ylims[1], ticks_freq))
    ax.set_xlabel(r'$\upsilon_0$ [km/s]', fontsize=BIGGER_SIZE, labelpad=10)
    ax.set_ylabel(r'$\mathscr{\gamma}$', fontsize=BIGGER_SIZE, labelpad=10)

    plot_literature(ax, authors2plot='all', s=700, fontsize_patch=19)
   
    if legend: leg2 = plot_colour_legend(ax, fontsize=MEDIUM_SIZE, loc='lower left',
                                         labels = ('Potential-dominated, no self-gravity',
                                                   'Potential-dominated + self-gravity',
                                                   'Supernovae feedback-dominated')
                                         )

    patches = []
    sigmas = np.arange(3,0,-1.0)
    
    if kind=='portions' and separate_clds:
        for case in data_dict:
            data = data_dict[case][:,:2]
            npts_cld = int(len(data)/2)
            for i in range(2):
                if case == 'pot230nSG': cld = [r'A$_0$',r'B$_0$']
                if case == 'pot230SGjeans': cld = ['A','B']
                if case == 'Fjeans248': cld = ['C','D']
                datai = data[i*npts_cld:(i+1)*npts_cld]
                meani = np.mean(datai, axis=0)
                stddevi = np.std(datai, axis=0)
                if log: print ('mean and stddev (v0, gamma) (dv0, dgamma) for case %s in kind %s, cloud %s:'%(case, kind, cld[i]), meani, stddevi)
                for sig in sigmas:
                    ax.add_patch(mpatches.Ellipse(meani, *(stddevi*sig), linewidth=1.5,
                                                  facecolor=colors_dict[scenarios[case]], edgecolor='k', alpha=0.8/sig))
                    ax.text(*meani, cld[i], ha='center', va='center')
    else:
        for case in data_dict:
            data = data_dict[case][:,:2]
            mean = np.mean(data, axis=0)
            stddev = np.std(data, axis=0)
            if log: print ('mean and stddev (v0, gamma) (dv0,dgamma) for case %s in kind %s:'%(case, kind), mean, stddev)
            for sig in sigmas:
                ax.add_patch(mpatches.Ellipse(mean, *(stddev*sig), linewidth=1.5,
                                              facecolor=colors_dict[scenarios[case]], edgecolor='k', alpha=1.0/sig))
    
    ax.text(0.02,0.915, kind_dict[kind], fontsize=MEDIUM_SIZE, transform=ax.transAxes)
    plot_pars_histogram(ax1, kind, data_dict, data_axis = 'gamma')
    plot_pars_histogram(ax2, kind, data_dict, data_axis = 'v0')

    output = 'PCA_histograms_%s_%s'%(units[0],kind)
    if separate_clds: output+='_sep'
    output+='.pdf'
    fig.savefig(output, dpi=500, bbox_inches='tight') 


def figure_time_evolution(kind, data_tag, ylims=(0.0,1.4), ticks_freq=[0.1,0.1,0.2], units=['jypxl'], legend_size=True, legend_colour=True):
    size_inches = (12.,4.)
    ratio = size_inches[0]/size_inches[1]
    fig = plt.figure(figsize=size_inches) #, ax = plt.subplots(ncols=3, figsize=(12,4), sharey=True)
    ax = [fig.add_axes([0.05+0.3*i,0.1,0.3,0.3*ratio]) for i in range(3)]
    ax[0].set_xlim(0.2,0.601)
    ax[1].set_xlim(0.1,0.801)
    ax[2].set_xlim(0.7,1.701)
    for i,axi in enumerate(ax): 
        axi.set_ylim(ylims)
        lims = axi.get_xlim()
        axi.set_xticks(np.arange(lims[0]+0.1, lims[1]-0.1, ticks_freq[i]))
        axi.set_yticks(np.arange(0.1, ylims[1], 0.2))
        axi.set_xlabel(r'$\upsilon_0$ [km/s]', fontsize=BIGGER_SIZE, labelpad=10)
        if i > 0: axi.tick_params(labelleft=False)

    ax[0].set_ylabel(r'$\mathscr{\gamma}$', fontsize=BIGGER_SIZE, labelpad=10)
    for axi in ax: plot_literature(axi, authors2plot=['Kolmogorov+1941', 'Kraichnan+1974', 'Larson+1981', 'Solomon+1981','Heyer+2004', 'Roman-Duval+2011', 'Bertram+2014', 'Hacar+2016'])
    
    cases = ['pot230nSG', 'pot230SGjeans', 'Fjeans248']

    arrow = mpatches.ArrowStyle.CurveFilledB(head_length=0.25, head_width=0.1)
    for i,case in enumerate(cases):
        data = np.array(data_tag[case]['data']).squeeze()[:,:2]
        time = np.array(data_tag[case]['time'])
        incl = np.array(data_tag[case]['orientation'])
        cloud = np.array(data_tag[case]['cloud'])
        
        unique_times = np.unique(time)
        unique_incs = np.unique(incl)
        unique_clouds = np.unique(cloud)
        
        for ti in range(1, len(unique_times)):
            x,y = data[time==unique_times[ti-1]].T
            xf,yf = data[time==unique_times[ti]].T
            dx = xf-x
            dy = yf-y

            for n in range(len(x)): 
                dr = np.array([dx[n],dy[n]])
                dr /= np.linalg.norm(dr)
                dr /= 50.
                ax[i].annotate("", xy=(xf[n], yf[n]), xytext=(x[n], y[n]),
                               arrowprops=dict(arrowstyle=arrow, color='k',linewidth=1.0))
                                               
    for i,case in enumerate(cases):
        for n in range (len(data_tag[case]['data'])):
            dv,gamma,dv_err,gamma_err = data_tag[case]['data'][n]
            marker = markers[data_tag[case]['orientation'][n]]
            size = size_markers(data_tag[case]['time'][n], marker)
            ax[i].scatter(dv,gamma, marker=marker, facecolor=colors_dict[scenarios[case]], edgecolor='k', linewidths=1.1, s=size, alpha=0.8, zorder=3)
            ax[i].errorbar(dv,gamma, xerr=dv_err, yerr=gamma_err, ecolor = colors_dict[scenarios[case]], linestyle='none', elinewidth=3, alpha=0.5, zorder=0)    
            cloud_name = data_tag[case]['cloud'][n]
            if case == 'pot230nSG': cloud_name += r'$_0$'
            for xi,yi in zip(dv,gamma): ax[i].text(xi-0e-3,yi-4.5e-3, cloud_name, weight='normal',fontsize=SMALL_SIZE-0.5, horizontalalignment='center', verticalalignment='center')


    if legend_size:
        ax1 = fig.add_axes([0.02,0.98,0.3,0.3])
        plot_sizeaxes(ax1, tick_params = dict(direction='in', left=False, labelleft=False, labelbottom=False, top=True, labeltop=True))
        ax1.xaxis.set_label_position('top')
    if legend_colour:
        leg2 = plot_colour_legend(ax[2], bbox_to_anchor=(0.96,1.0), bbox_transform=fig.transFigure,  
                                  labels = ('Potential-dominated, no self-gravity',
                                            'Potential-dominated + self-gravity',
                                            'Supernovae feedback-dominated'),
                                  fontsize = MEDIUM_SIZE - 3
                                  )
    fig.savefig('PCA_timesplit_%s_%s.pdf'%(units[0],kind), dpi=500, bbox_inches='tight') 
    return 0


mixed = figure_main('mixed')
cloud = figure_main('cloud')

tau_mixed = figure_main('mixed', units = ['tau'])
xlims = (-0.5,2.5)
ylims = (-1.5,2.5)
portions = figure_main('portions', xlims, ylims, ticks_freq=0.3)

figure_histograms('mixed', mixed[0], legend=True, log=True)
figure_histograms('cloud', cloud[0], log=True)
xlims = (0.0,2.2)
ylims = (-0.5,1.5)
figure_histograms('portions', portions[0], separate_clds=False, log=True) #, xlims, ylims, ticks_freq=0.3, pdf=True*0)
figure_histograms('portions', portions[0], separate_clds=True, log=True) #, xlims, ylims, ticks_freq=0.3, pdf=True*0)

figure_time_evolution('mixed', mixed[1])
figure_time_evolution('cloud', cloud[1], legend_size=False, legend_colour=False)
plt.show()

#Missing: pdf for portions' case
