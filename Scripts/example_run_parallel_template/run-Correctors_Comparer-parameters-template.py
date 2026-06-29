#!/usr/bin/env python
# coding: utf-8

import sys, os
sys.path.insert(0,'<enter_your_lightkurve_path>')
origin_dir='<enter_your_SysCoCoPy_path>'
sys.path.append(origin_dir+'/SysCoCoPy')
os.chdir(origin_dir)

import numpy as np
import time

from Source.Correctors_Comparer import Correctors_Comparer

correctors=['RCQ','PLD','CBV']

run = 'template'
targets = 'targets_{}.csv'.format(run)
save_name = 'saved_{}'.format(run)
#select a case or choose None or 'all'
select_case = sys.argv[1]
first_cindex = 1
ncases_per_bin = 2
#SPOC outliers are removed using the bitmask
#This is for using the lk remove_outliers method
remove_outliers = False
propagate_errors = True

overwrite_ok = True
download = False
metrics_only = False

plot_type = 'joined_panels'
binned_lcs = True
bin_size = 0.02
diag_pvar = False
corr_diags = False
#This is deprecated; all variation parameters are plotted at this time
diag_parName = 'spline_n_knots'

parameters = {
    'RCQ' : {
        'params_to_var'  : {
            'spline_n_knots' : [6,12,24,48],
            'pca_components' : [3,6,9]
        },
        'ntimes_nanstd'  : 10,
        'add_bkg_flag'   : True,
        'bkg_adjust'     : 'median',
        'color'          : 'blue',
        'colormap'       : 'Blues'
    },
    'PLD' : {
        'params_to_var'  : {
            'spline_n_knots'    : [6,12,24,48],
            'pld_aperture_mask' : ['pipeline','threshold'],
            'pca_components'    : [8,16,32]
        },
        'add_bkg_flag'   : True,
        'aperture_mask'  : 'pipeline',
        'bkg_adjust'     : 'median',
        'color'          : 'blueviolet',
        'colormap'       : 'Purples'
    },
    'CBV'  : {
        'params_to_var'  : {
            'cbv_type'       : [['MultiScale.1', 'MultiScale.2', 'MultiScale.3','Spike'],['SingleScale','Spike']],
            'cbv_indices'    : [[np.arange(1,9), np.arange(1,9), np.arange(1,9), 'ALL'],[np.arange(1,9), 'ALL']],
            'alpha_reg'      : [1e-4,1e-2,1],
            'spline_flag'    : [False],
            'spline_n_knots' : [0]
        },
        'add_bkg_flag'   : False,
        'auto_reg'       : False,
        'color'          : 'green',
        'colormap'       : 'Greens'
    },
    'PDC'  : {
        'color'          : 'red',
        'colormap'       : 'Reds'
    }
}

comparer = Correctors_Comparer(targets=targets,select_case=select_case,
                               first_cindex=first_cindex,
                               ncases_per_bin=ncases_per_bin,
                               remove_outliers=remove_outliers,
                               propagate_errors=propagate_errors,
                               diag_pvar=diag_pvar,diag_parName=diag_parName,
                               correctors=correctors,parameters=parameters)

start = time.time()
comparer.compare(save_name=save_name,overwrite_ok=overwrite_ok,download=download,
                 metrics_only=metrics_only,corr_diags=corr_diags)
end = time.time()
print('For selected case: ',select_case)
print('The elapsed time of the compare function is: ', end - start)
