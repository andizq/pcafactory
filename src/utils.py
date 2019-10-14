import os
import numpy as np

home = os.environ['HOME']
cwd = os.getcwd()

unit_dict = {'jypxl': 'Jy/pxl',
             'tau': r'$\tau$',
             'kelvin': 'K'}

rt_dict = {'Lime-arepo': 'LTE',
           'Polaris-arepo': 'LVG',
           'Lime-arepo-nonLTE': 'nonLTE'}

case_dict = {'Fjeans': 'Self-Gravity + Mixed Supernovae', 
             'nSG': 'No Self-Gravity + Random Supernovae', 
             'SGjeans': 'Self-Gravity + Random Supernovae'}

get_rt_mode = lambda path=cwd: rt_dict[path.split('RT-Analysis/')[1].split('/')[0]] #Gets RT mode based on root directory

literature_dict = {'Kolmogorov+1941': {'kwargs': {'linestyle': '--', 'color': 'blue'}, 'v0': None, 'gamma': 1/3., 'dv0': None, 'dgamma': None, 'kind': 'line'},
                   'Kraichnan+1974': {'kwargs': {'linestyle': '--', 'color': 'red'}, 'v0': None, 'gamma': 1/2., 'dv0': None, 'dgamma': None, 'kind': 'line'},
                   'Larson+1981': {'kwargs': {'marker': r'$\mathcircled{L}$'}, 'v0': 1.1, 'gamma': 0.38, 'dv0': None, 'dgamma': None, 'kind': 'point'},
                   'Solomon+1981': {'kwargs': {'marker': r'$\mathcircled{S}$'}, 'v0': 1.0, 'gamma': 0.5, 'dv0': 0.1, 'dgamma': 0.05, 'kind': 'point'},
                   'Heyer+1997(a)': {'kwargs': {}, 'marker_label': r'1a', 'v0': 1.23, 'gamma': 0.24, 'dv0': 0.08, 'dgamma': 0.10, 'kind': 'patch'},
                   'Heyer+1997(b)': {'kwargs': {}, 'marker_label': r'1b', 'v0': 0.78, 'gamma': 0.42, 'dv0': 0.05, 'dgamma': 0.09, 'kind': 'patch'},
                   'Heyer+2004': {'kwargs': {'marker': r'$\mathcircled{H}$'}, 'v0': 0.87, 'gamma': 0.57, 'dv0': 0.02, 'dgamma': 0.07, 'kind': 'point'},
                   'Heyer+2006(a)': {'kwargs': {}, 'marker_label': r'3a', 'v0': 0.73, 'gamma': 0.71, 'dv0': 0.03, 'dgamma': 0.12, 'kind': 'patch'},
                   'Heyer+2006(b)': {'kwargs': {}, 'marker_label': r'3b', 'v0': 1.00, 'gamma': 0.79, 'dv0': 0.04, 'dgamma': 0.04, 'kind': 'patch'},
                   'Heyer+2006(c)': {'kwargs': {}, 'marker_label': r'3c', 'v0': 0.70, 'gamma': 0.59, 'dv0': 0.03, 'dgamma': 0.14, 'kind': 'patch'},
                   'Bolatto+2008': {'kwargs': {'marker': r'$\mathcircled{4}$'}, 'v0': 0.76, 'gamma': 0.60, 'dv0': 0.27, 'dgamma': 0.10, 'kind': 'point'},
                   'Federrath+2010(a)': {'kwargs': {'linestyle': '-', 'color': 'darkred'}, 'v0': None, 'gamma': 0.59, 'dv0': None, 'dgamma': 0.13, 'kind': 'line'},
                   'Federrath+2010(b)': {'kwargs': {'linestyle': '-', 'color': 'dodgerblue'}, 'v0': None, 'gamma': 0.74, 'dv0': None, 'dgamma': 0.19, 'kind': 'line'},
                   'Klessen+2010': {'kwargs': {'marker': r'$\mathcircled{5}$'}, 'v0': 0.8, 'gamma': 0.5, 'dv0': None, 'dgamma': None, 'kind': 'point'},
                   'Roman-Duval+2011': {'kwargs': {'linestyle': '-', 'color': 'green'}, 'v0': None, 'gamma': 0.53, 'dv0': None, 'dgamma': 0.35, 'kind': 'line'},
                   'Bertram+2014': {'kwargs': {'linestyle': '-', 'color': 'indigo'}, 'v0': None, 'gamma': 0.93, 'dv0': None, 'dgamma': 0.14, 'kind': 'line'},
                   'Hacar+2016': {'kwargs': {'marker': r'$\mathcircled{M}$'}, 'v0': 0.66, 'gamma': 0.58, 'dv0': None, 'dgamma': None, 'kind': 'point'},
                   'Padoan+2017': {'kwargs': {'marker': r'$\mathcircled{6}$'}, 'v0': 0.82, 'gamma': 0.5, 'dv0': None, 'dgamma': 0.1, 'kind': 'point'},
                   'Traficante+2018b': {'kwargs': {'linestyle': '-', 'color': 'darkgoldenrod'}, 'v0': None, 'gamma': 0.09, 'dv0': None, 'dgamma': 0.04, 'kind': 'line'},
                   }

def get_cloud_history(path=cwd, add=None, arepo_step=0.2, ret_label=False):
    #arepo_step Myr
    #if path is None: path = cwd
    cloud = path.split('cld')[1][0]
    rt_mode = get_rt_mode(path)
    
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


def alpha_to_beta(alpha, dalpha):
    alpha = np.asarray(alpha)
    dalpha = np.asarray(dalpha)
    x = 0.20
    y = 2.99
    dx = 0.05
    dy = 0.09
    beta = x+y*alpha
    dbeta = dx + alpha*dy + y*dalpha
    return (beta, dbeta)
    
def alpha_to_gamma(alpha, dalpha, ret_beta=False):
    (beta, dbeta) = alpha_to_beta(alpha, dalpha) 
    gamma = 0.5*(beta-1)
    dgamma = 0.5*dbeta
    if ret_beta: return (gamma, dgamma), (beta, dbeta)
    else: return (gamma, dgamma)
