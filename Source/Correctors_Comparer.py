"""
File: SysCoCoPy/Correctors_Comparer.py
Author: David Rapetti
Date: 17 Feb 2022

Description: Compares the systematic corrections of various 
             Lightkurve correctors using a series of validation 
             metrics.
"""

import lightkurve as lk
import numpy as np
import numpy.ma as ma
import sys, gc, json
import dill as pickle
import statistics
import copy
import matplotlib.pyplot as plt
import os, tempfile
from os import walk
from os import makedirs
from matplotlib import cm
from matplotlib import colors as matcolors
from astropy import units as u
from astropy.units import cds, Quantity
from astropy.io import fits
from astropy.utils.masked import core
from astropy.table import setdiff, Table
from itertools import product,groupby
from lightkurve.correctors import RegressionCorrector, \
    PLDCorrector, CBVCorrector
from lightkurve.correctors.designmatrix import DesignMatrix, \
    create_spline_matrix, DesignMatrixCollection
#from Util.Target_Lists.Targets_file_reader import read_target_file
from Util.Target_Lists.Targets_file_reader import read_target_cases, \
    download_target_files
from Util.Quaternions_Design_Matrix import *
#TODO: Move metrics calculations to a new class Metrics_Provider
from lightkurve.correctors.metrics import overfit_metric_lombscargle, \
    underfit_metric_neighbors

default_parameters = {
    'RCQ' : {
        'params_to_var'  : {
            'spline_n_knots' : [80],
            'pca_components' : [3]
        },
        'ntimes_nanstd'  : 10,
        'add_bkg_flag'   : True,
        'bkg_adjust'     : 'median',
        'color'          : 'blue',
        'colormap'       : 'Blues'
    },
    'PLD' : {
        'params_to_var'  : {
            'spline_n_knots'    : [80],
            'pld_aperture_mask' : ['pipeline'],
            'pca_components'    : [3]
        },
        'add_bkg_flag'   : True,
        'aperture_mask'  : 'pipeline',
        'bkg_adjust'     : 'median',
        'color'          : 'blueviolet',
        'colormap'       : 'Purples'
    },
    'CBV'  : {
        'params_to_var'  : {
            'cbv_type'       : [['SingleScale','Spike']],
            'cbv_indices'    : [[np.arange(1,9), 'ALL']],
            'alpha_reg'      : [1e-4],
            'spline_flag'    : [True],
            'spline_n_knots' : [80]
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

"""
If specific quality and transit masks are needed, the following 
quantities would needed to be provided for each target.
"""

#quality_bitmask = 16575   # from JT
quality_bitmask = 16575+512 # from JT+filter out SPOC outliers from lc/tpf
##quality_bitmask = ( 4 | 16 | 32 ) # from H21

#transit_time = 0
#period = 0
#duration = 0

minutes_cadence=2
savgol_polyorder=2
#savgol_windows=[td*ratio_window_duration for td in transit_durations]
#transit_durations=np.array([[30],[60],[120],[720],[3600]])
transit_durations=np.array([[15],[30],[60],[80],[100]])
ratio_window_duration=5
savgol_windows=np.array(5*[[transit_durations[4,0]*ratio_window_duration]])
#savgol_windows=np.array(transit_durations*ratio_window_duration)

minutes_list=[]
for transit_duration in transit_durations:
    minutes=int(transit_duration*minutes_cadence)
    minutes_list.append(minutes)
                
dir_to_files='data_files/saved_output/'
available_correctors = list(default_parameters.keys())

#number of samples for the overfit metric
n_samples = 10

class Correctors_Comparer(RegressionCorrector):
    """
    0) PDC   : SPOC PDCSAP from 2-min TPFs serve as reference results (thus, 
               included always by default)
    
    Correctors starting from SAP + background flux:
    
    1) RCQ : RegressionCorrector following Hedges et al, AJ 162, 54 (2021), 
    hereafter H21, with a design matrix formed by
        a) a spline for stellar variability
        b) PCA components for the background light
        c) spacecraft quaternions for TESS jitter

    2) PLD   : PLDCorrector using a desgin matrix formed by
        a) a spline for stellar variability
        b) PCA components for the background light
        c) Pixel Level Decorrelation (PLD) for instrument jitter
        
    Corrector starting from SAP flux:
    
    3) CBV : CBVCorrector plus a spline
    """
    
    def __init__(self, targets, correctors, select_case=None, first_cindex=1, 
                 ncases_per_bin=1, remove_outliers=False, diag_pvar=False,
                 diag_parName=None,propagate_errors=False, 
                 parameters=None):
        """
        This constructor collects input for subsequent corrector comparisons:
            a) targets/sectors file name, 
            b) correctors names, 
            c) parameter values
        
        If select_case is None (by default), it proceeds with the case list. If 
        a case is chosen, it runs only that case (e.g., for testing purposes).
        
        Also, the case list can be a sublist of the targets/sectors file 
        starting with:
            first_cindex -- which is 1 by default.
        
        And the number of cases in one bin, which might change memory usage, 
            ncases_per_bin -- is set to 1 by default.
            
        diag_pvar    : set True to use the diagnose_paramvar method
                       to see the results from a corrector parameter range
        diag_parName : for diag_pvar, a parameter to diagnose can be chosen
        """
        
        self.targets = targets
        self.select_case = select_case
        self.first_cindex = first_cindex-1
        self.ncases_per_bin = ncases_per_bin
        self.correctors = correctors
        self.parameters = parameters
        self.cbv_dir = None
        self.remove_outliers = remove_outliers
        #currently only for RCQ and PLD
        self.propagate_errors=propagate_errors
        self.diag_pvar = diag_pvar
        self.diag_parName = diag_parName
        
        self.paramvar_arr = {}
        self.param_names = {}
        
        self.end_of_bin = self.first_cindex + self.ncases_per_bin
        
        self.valid_corrs(correctors)
        
        if self.parameters==None:
            self.parameters = default_parameters    
        
        if self.parameters['RCQ']['add_bkg_flag']==False:
            raise ValueError('RCQ requires add_bkg_flag to be True.')
        if self.parameters['CBV']['add_bkg_flag']==True:
            raise ValueError('CBV requires add_bkg_flag to be False.')
        
        return
    
    def valid_corrs(self,correctors):
        """
        Raises an error if any corrector in the input 
        list of correctors is not available
        """

        valid_corrs = all(corrector in available_correctors 
                           for corrector in correctors)
        
        if valid_corrs == False:
            raise Exception('Correctors must be in the list available')
            
        return
    
    def load_saved_input(self,save_name):
        """
        Loads previously corrected light curve objects
        from a saved file name.
        """
        
        path_to_files = dir_to_files+save_name+'/'
        path_to_pkl_file = path_to_files+'{}.pkl'.format(save_name)
        #TODO: MOVE LISTS TO PLOTTING
        #path_to_sap_files = path_to_files+'fits_files/sap/' 
        
        with open(path_to_pkl_file, 'rb') as pickle_file:
            #TODO: MOVE LISTS TO PLOTTING
            #Self.corrected_lc_list = pickle.load(plots_file)
            #self.metrics_list = pickle.load(plots_file)
            self.spoc_lc_filenames = pickle.load(pickle_file)
            self.tpf_filenames = pickle.load(pickle_file)
            self.quat_list = pickle.load(pickle_file)
            self.cbv_dir = pickle.load(pickle_file)
            self.cases_list = pickle.load(pickle_file)
            #TODO: ELIMINATE METRICS LIST AND USE INSTEAD METRICS JSON FILES
            #self.metrics_list = pickle.load(pickle_file)

#         TODO: MOVE LISTS TO PLOTTING
#         filenames = next(walk(path_to_sap_files), (None, None, []))[2]
        
#         self.sap_lc_list = []
#         for i in np.arange(len(filenames)):
#            self.sap_lc_list.append('')
        
#         for filename in filenames:
#            if filename.endswith('.fits'):
#                sap_lc=lk.read('{}{}'.format(path_to_sap_files,filename))
#                cindex=(filename.removesuffix('.fits').split('-')[-1])
#                index=int(cindex)-1
#                if index >= initial_index and index <= final_index:
#                    self.sap_lc_list[index]=sap_lc
      
        return
    
    def load_saved_metrics(self,save_name):
        """
        Loads previously calculated metrics from a saved file name. 
        """
        
        self.metrics_list=[]
        #loads metrics dictionaries from json files into the metrics list
        path_to_files = dir_to_files+save_name+'/'
        path_to_json = path_to_files+'json_files/metrics'
        for case in self.cases_list:
            tic = case['TIC']
            sector = case['Sector']
            case_idx = case['Case_Index']
            metrics_name = \
            'metrics_tic-{}-sector-{}-cindex-{}'.format(
                tic, sector, case_idx)
            
            with open('{}/{}.json'.format(
                path_to_json,metrics_name), 'r') as openfile:
                metrics_dict = json.load(openfile)
            self.metrics_list.append(metrics_dict)
            
        return
    
    def compare(self,save_name=None,overwrite_ok=False,download=True,
                metrics_only=False,corr_diags=False):
        """
        Reads in the required tpf, lf, and quat filenames for the case list.
        
        Runs each of the correctors on the selected target cases using 
        the chosen set of parameters for the corrector and saves sap, 
        pdc (symlinks), and corrected lcs in fits files. 
        
        Calculates and saves metrics in a pkl file.
        
        Optional parameters:
        
        save_name    : sets the name of the directory storing fits files
                       of light curves. The same name is used for the pkl file 
                       storing the cases and metrics list objects
        overwrite_ok : set True if you are ok with overwriting files and
                       directories and want to avoid questions about it
        download     : set False if there is no need for downloading data files
        metrics_only : set True if corrected fits files are already calculated 
                       and only metrics are needed
        corr_diags   : set True to use the diagnose methods from the correctors
        """
        
        if save_name == None:
            save_name=input('Enter name of saving directory: ')
        
        self.corr_diags = corr_diags
        
        path_to_pkl_file = '{}{}/{}.pkl'.format(
            dir_to_files,save_name,save_name)
        path_to_pkl_file, path_to_dir = self.set_file_dir(
            path_to_pkl_file,save_name,overwrite_ok)

        if download == True:
            self.cases_list = read_target_cases(self.targets)
            #TODO: TESTING FILENAMES
            # self.spoc_lc_list, self.tpf_list, self.quat_list, \
            self.spoc_lc_filenames, self.tpf_filenames, self.quat_list, \
            self.cbv_dir = download_target_files(
                self.cases_list, quality_bitmask)
            with open(path_to_pkl_file, 'wb') as pickle_file:
                #TODO: MOVE LISTS TO PLOTTING
                #pickle.dump(self.corrected_lc_list, plots_file)
                pickle.dump(self.spoc_lc_filenames, pickle_file) 
                pickle.dump(self.tpf_filenames, pickle_file)
                pickle.dump(self.quat_list, pickle_file)
                pickle.dump(self.cbv_dir, pickle_file)
                pickle.dump(self.cases_list, pickle_file)
        else:
            self.load_saved_input(save_name=save_name)
        
        #TODO: MOVE LISTS TO PLOTTING
        # if cindex==0:
        #     self.corrected_lc_list = []
        #     self.sap_lc_list = []
        #     self.metrics_list = []
        # else:
        #     self.load_saved_lcs(cindex,self.end_of_bin,save_name=save_name)
        
        #TODO: ELIMINATE METRICS LIST AND USE INSTEAD METRICS JSON FILES
        #self.metrics_list = []
        
        total_num_cases = len(self.cases_list)
        num_cases = total_num_cases - self.first_cindex
        num_full_bins = int(num_cases / self.ncases_per_bin)
        reminder_cases = num_cases % self.ncases_per_bin

        if self.select_case!=None:
            self.compare_bin(save_name,overwrite_ok,path_to_dir,
                             self.select_case-1,self.select_case,metrics_only)
            return
            
        for bin_idx in np.arange(num_full_bins):
            initial_cindex = self.first_cindex + bin_idx * self.ncases_per_bin
            final_cindex = initial_cindex + self.ncases_per_bin
            self.compare_bin(save_name,overwrite_ok,path_to_dir,initial_cindex,
                             final_cindex,metrics_only)
            gc.collect()
        
        if reminder_cases != 0:
            self.compare_bin(save_name,overwrite_ok,path_to_dir,
                             total_num_cases-reminder_cases,total_num_cases,
                             metrics_only)
            gc.collect()
        
        # #if save_name != None:
        # with open(path_to_pkl_file, 'wb') as pickle_file:
        #     #TODO: MOVE LISTS TO PLOTTING
        #     #pickle.dump(self.corrected_lc_list, plots_file)
        #     pickle.dump(self.metrics_list, pickle_file)
        #     #TODO: ELIMINATE METRICS LIST AND USE INSTEAD METRICS JSON FILES
        
        return

    def compare_bin(self,save_name,overwrite_ok,
                    path_to_dir,first_cidx,last_cidx,metrics_only):
        """
        Runs one bin of the compare method - see further details there.
        """

        cindex = first_cidx
        for case in self.cases_list[first_cidx:last_cidx]:
            tic = case['TIC']
            sector = case['Sector']
            case_idx = case['Case_Index']
            
            quat_file = self.quat_list[cindex]
            
            #TODO: TESTING FILENAMES
            #reads tpfs and spoc lcs
            tpf = lk.read('{}'.format(self.tpf_filenames[cindex]),
                          quality_bitmask=quality_bitmask)
            spoc_lc = lk.read('{}'.format(self.spoc_lc_filenames[cindex]),
                              quality_bitmask=quality_bitmask)

            #tpf = self.tpf_list[cindex]
            tpf_nb, lc_nb = self.prepare_lc(tpf, add_bkg=False)
            tpf_b, lc_b, bkg_pixels = self.prepare_lc(tpf, add_bkg=True)
            
            #cadence masks based on PDCSAP light curves
            cadence_mask = np.empty(len(lc_nb), dtype=bool)
            
            for idx in np.arange(len(lc_nb)):
                #TODO: TESTING FILENAMES
                if ma.is_masked(spoc_lc.flux[idx]):
                #if ma.is_masked(self.spoc_lc_list[cindex].flux[idx]):
                    cadence_mask[idx] = False
                else:
                    cadence_mask[idx] = True
            
            sap_lc_name = 'sap_lc_tic-{}-sector-{}-cindex-{}'.format(
                    tic,sector,case_idx)
            path_to_fits=path_to_dir+'fits_files/sap'
            makedirs(path_to_fits,exist_ok=True)
            
            #adjusts SAP light curves
            
            #TODO: TESTING FILENAMES
            #sap_lc = copy.deepcopy(self.spoc_lc_list[cindex])
            
            #sap_lc = copy.deepcopy(spoc_lc)
            #sap_lc.flux = sap_lc['sap_flux']
            #sap_lc.flux_err = sap_lc['sap_flux_err']
            #instead of the above, for now we use to_lightcurve to obtain 
            #comparable errors (i.e. assuming no correlation between 
            #the background errors) to those of the corrected light curves;
            #except for the calculation of the PDC overfit metric
            #TODO: this could be updated in the future;
            #see the prepare_lc and to_spoc_sap_flux_err methods
            sap_lc = tpf.to_lightcurve(method='sap')
            #applying flux fraction and crowding for comparison purposes with
            #the corrrected light curves
            sap_lc = self.flux_adjust(tpf_nb,sap_lc)
            
            if metrics_only==False:
                #saves SAP lcs to fits files
                sap_lc.to_fits(path='{}/{}.fits'.format(
                    path_to_fits,sap_lc_name), overwrite=True)
                #TODO: MOVE LISTS TO PLOTTING
                #self.sap_lc_list.append(sap_lc)
            
                #makes symlinks for the PDC lc fits files
                spoc_lc_name = 'pdc_lc_tic-{}-sector-{}-cindex-{}'.format(
                    tic,sector,case_idx)
                path_to_fits=path_to_dir+'fits_files/pdc'
                makedirs(path_to_fits,exist_ok=True)
                source_name=self.spoc_lc_filenames[cindex]
                link_name='{}/{}.fits'.format(path_to_fits,spoc_lc_name)
                self.symlink(source_name,link_name,overwrite=True)
            
                gc.collect()
                        
            correctors_dict={}
            metrics_dict={}
            for corrector in self.correctors:
                corr_lc_name = \
                'corrected_lc_corrector-{}-tic-{}-sector-{}-cindex-{}'.format(
                    corrector, tic, sector, case_idx)
                path_to_fits = path_to_dir+'fits_files/corrected'
                
                if metrics_only==False:
                    if corrector=='RCQ':
                        #TODO: Potentially extending corr_lc to a list 
                        #for other parameters
                        corr_lc = self.RCQ(cindex,tpf_b,lc_b,
                                           bkg_pixels,quat_file) 
                    if corrector == 'PLD':
                        if self.parameters['PLD']['add_bkg_flag']==True:
                            tpf_in, bkg_pixels = tpf_b, bkg_pixels
                        else:
                            tpf_in, bkg_pixels = tpf_nb, None
                        corr_lc = self.PLD(cindex,tpf_in, bkg_pixels)  
                    if corrector == 'CBV':
                        corr_lc = self.CBV(cindex,cadence_mask, tpf_nb, lc_nb)
                    
#                     if self.parameters[corrector]['add_bkg_flag']==True:
#                         print('here',corr_lc)
#                         corr_lc=self.flux_level_adjust(tpf_b,corr_lc,bkg_pixels)
#                         corr_lc=self.flux_adjust(tpf_b,corr_lc)
#                         print('there',corr_lc)
#                     else:
#                         corr_lc=self.flux_adjust(tpf_nb,corr_lc)
                    
                    correctors_dict[corrector] = corr_lc
                    
                    # self.save_to_fits_file(path_to_fits,corrector,corr_lc,
                    #                        corr_lc_name)
                    
                if metrics_only==True:
                    corr_lc = lk.read('{}/{}.fits'.format(
                        path_to_fits,corr_lc_name),
                                      quality_bitmask=quality_bitmask)
                
                metrics_col={}
                hm_dict = {}
                flux_columns=[x for x in corr_lc.colnames if 'flx_parvar' in x]
                flux_err_columns=\
                [x for x in corr_lc.colnames if 'flx_err_parvar' in x]
                
                for flux_column,flux_err_column in zip(flux_columns,
                                                       flux_err_columns):
                    col_corr_lc=corr_lc.select_flux(flux_column,flux_err_column)
                    
                    #adding cadence_mask of PDC for the metrics of PLD, RCQ, SAP
                    if corrector == 'RCQ' or corrector == 'PLD':
                        metrics_col[flux_column] = \
                        self.new_metrics_calculate(
                            #TODO: TESTING FILENAMES
                            #self.spoc_lc_list[cindex],sap_lc)
                    #TODO: new version of the metrics_calculate method to be incoporated 
                    #into the rest of metrics methods
                            #corr_lc[cadence_mask],lc_b,spoc_lc,sap_lc[cadence_mask])
                            col_corr_lc[cadence_mask],sap_lc[cadence_mask])
                        
                    #adding cadence_mask of PDC for the metrics of SAP
                    if corrector == 'CBV':
                        metrics_col[flux_column] = \
                        self.new_metrics_calculate(
                            #TODO: TESTING FILENAMES
                            #corr_lc,lc_nb,self.spoc_lc_list[cindex],sap_lc)
                    #TODO: new version of the metrics_calculate method to be incoporated 
                    #into the rest of metrics methods 
                            #corr_lc,lc_nb,spoc_lc,sap_lc[cadence_mask])
                            col_corr_lc,sap_lc[cadence_mask])
                    
                    sgcdpp=\
                    [x[1] for x in metrics_col[flux_column].items() \
                     if 'sgCDPP' in x[0]]
                    #TODO: Potentially include the overfit in the future to be
                    #used as a goodness-of-fit metric in combination with CDPP
                    
                    hm = statistics.harmonic_mean(sgcdpp)
                    #metrics_col[flux_column]['sgCDPP_harmonic_mean'] = hm
                    hm_dict[flux_column] = hm *\
                        u.dimensionless_unscaled.to(cds.ppm)
                    
                #metrics_dict[corrector] = metrics_col

                #reorderding the flux+flux_err columns by increasing hm metric
                sorted_flux_columns = sorted(hm_dict, key=hm_dict.get)
                sorted_flux_err_columns = [x.replace(
                    'flx','flx_err') for x in sorted_flux_columns]
                
                plength=len(self.paramvar_arr[corrector])*2
                orig_cols = corr_lc.colnames
                del orig_cols[-plength:]
                parvar_sorted_cols = self.flatten2list(
                    list(zip(sorted_flux_columns,sorted_flux_err_columns)))
                
                corr_lc_sorted = copy.deepcopy(corr_lc)
                corr_lc_sorted.keep_columns(orig_cols)
                paramvar_list = []
                for col in parvar_sorted_cols:
                    corr_lc_sorted.add_column(corr_lc[col])
                    if 'flx_parvar_' in col:
                        idx = int(col.partition('flx_parvar_')[2])
                        paramvar_list.append(np.append(
                            self.paramvar_arr[corrector][idx],
                            #'  | '+str(round(hm_dict[col]))))
                            round(hm_dict[col])))
                self.paramvar_arr[corrector] = np.array(paramvar_list)
                
                min_flux_column = parvar_sorted_cols[0]
                min_flux_err_column = parvar_sorted_cols[1]
                
                min_corr_lc=corr_lc_sorted.select_flux(min_flux_column,
                                                       min_flux_err_column)
                
                #adding cadence_mask of PDC for the metrics of PLD, RCQ, SAP
                if corrector == 'RCQ' or corrector == 'PLD':
                    metrics_dict[corrector] = \
                    self.metrics_calculate(
                        #TODO: TESTING FILENAMES
                        #self.spoc_lc_list[cindex],sap_lc)
                        min_corr_lc[cadence_mask],lc_b,spoc_lc,
                        sap_lc[cadence_mask])
                        
                #adding cadence_mask of PDC for the metrics of SAP
                if corrector == 'CBV':
                    metrics_dict[corrector] =  \
                    self.metrics_calculate(
                        #TODO: TESTING FILENAMES
                        #corr_lc,lc_nb,self.spoc_lc_list[cindex],sap_lc)
                        min_corr_lc,lc_nb,spoc_lc,sap_lc[cadence_mask])
                
                makedirs(path_to_fits,exist_ok=True)
                path='{}/{}.fits'.format(path_to_fits,corr_lc_name)
                
                self.save_to_fits_file(path,corrector,min_corr_lc,
                                       corr_lc_name)
                
                if self.diag_pvar==True:
                    self.diagnose_paramvar(corrector,cindex,min_corr_lc,
                                           save_name,overwrite_ok,path=path)
                
            #TODO: MOVE LISTS TO PLOTTING
            # self.corrected_lc_list.append(correctors_dict)
            #TODO: ELIMINATE METRICS LIST AND USE INSTEAD METRICS JSON FILES
            #self.metrics_list.append(metrics_dict)

            # #saves metrics dictionaries to json files
            # metrics_name = \
            # 'metrics_tic-{}-sector-{}-cindex-{}'.format(
            #     tic, sector, case_idx)
            # path_to_json = path_to_dir+'json_files/metrics'
            # makedirs(path_to_json,exist_ok=True)
            # json_object = json.dumps(metrics_dict, indent = 4)
            # with open("{}/{}.json".format(
            #     path_to_json,metrics_name), "w") as outfile:
            #     outfile.write(json_object)
            
            self.save_metrics_to_json(save_name,overwrite_ok,
                                      metrics_dict,tic,sector,case_idx)
            
            cindex += 1
            
            gc.collect()

        return
    
    def save_to_fits_file(self,path,corrector,corr_lc,corr_lc_name):
        """
        Saves corrected lcs to fits files
        """
        
        #TODO: Extend for other correctors beyond PLD
        paramvar_arr = self.paramvar_arr[corrector]
        param_names = self.param_names[corrector]
        
        columns=corr_lc.colnames
        plength=len(paramvar_arr)*2
        paramcol_dict={}
        for col in columns[-plength:]:
            paramvar_column=np.array(corr_lc.columns[col])
            paramcol_dict[col]=paramvar_column
        
        corr_lc.to_fits(path=path,overwrite=True,
                        HLSPNAME='SysCoCoPy',
                        #HLSPLEAD='TESS SPOC pipeline team',
                        #CITATION='SURNAMEYEAR',
                        **paramcol_dict)
        
        #additional binary table extension for the parameter variation values
        with fits.open(path) as hdu:
            ncol=len(param_names)
            coldef_list=[]
            
            #adding the harmonic mean metric in the first column
            colT=paramvar_arr.transpose()[-1]    
            coldef=fits.Column(name='HM of sgCDPPs',
                                format='J',array=colT)
            coldef_list.append(coldef)
            
            for idx,col in enumerate(columns[-ncol:]):
                colT=paramvar_arr.transpose()[idx]
                
                #to avoid 'object' type for the fits file
                coltype=type(colT[0])
                colTT=colT.astype(coltype)
                
                name=param_names[idx]
                if isinstance(colT[0],int)==True:
                    coldef=fits.Column(name=name,format='J',array=colTT)
                elif isinstance(colT[0],float)==True:
                    coldef=fits.Column(name=name,format='E',array=colTT)    
                elif isinstance(colT[0],str)==True:
                    coldef=fits.Column(name=name,format='10A',array=colTT)
                
                elif isinstance(colT[0],list)==True:
                    arr_col = [str(x) for x in colTT]
                    len_col = [len(x) for x in arr_col]
                    max_len_col = np.sort(len_col)[-1]
                    coldef=fits.Column(name=name,format='{}A'.format(
                        max_len_col),array=arr_col)
                
                coldef_list.append(coldef)
            
            coldefs = fits.ColDefs(coldef_list)
            hdu.append(
                fits.BinTableHDU.from_columns(
                    coldefs,name='{} PARAMVAR'.format(corrector)))
            
            hdu.writeto(path, overwrite=True)
        
        return

    def set_file_dir(self,path_to_file,save_name,overwrite_ok):
        """
        Checks if the file exists and, if so, asks if to overwrite, and 
        if not, asks for a new file name.
        
        It also creates a directory to store files using the same name.
        """
        
        from os.path import exists
        
        file_exists = exists(path_to_file)
        
        path_to_dir = path_to_file.removesuffix('{}.pkl'.format(save_name))        
        dir_exists = exists(path_to_dir)
        
        if overwrite_ok == False:
            if file_exists == True:
                fcheck=input('The selected file {} already exists.\n'\
                             'Do you want to ovewrite it? (y/n) '.format(
                    path_to_file))
                if fcheck != 'y':
                    path_to_file=input(
                        'Enter a different file name '\
                        '(including path and extenstion): ')
                
        if dir_exists == False:
            makedirs(path_to_dir)
        else:
            if overwrite_ok == False:
                dcheck=input('The selected directory {} already exists.\n'\
                             'Do you want to ovewrite it? (y/n) '.format(
                    path_to_dir))
                if dcheck != 'y':
                    path_to_dir=input('Enter a different '\
                                       'directory name (including path): ')
                    makedirs(path_to_dir,exist_ok=False)
        
        return path_to_file, path_to_dir

    def symlink(self, target, link_name, overwrite=False):
        '''
        Create a symbolic link named link_name pointing to target.
        If link_name exists then FileExistsError is raised, unless 
        overwrite=True. When trying to overwrite a directory, 
        IsADirectoryError is raised.
        '''
        
        if not overwrite:
            os.symlink(target, link_name)
            return

        # os.replace() may fail if files are on different filesystems
        link_dir = os.path.dirname(link_name)

        # Create link to target with temporary filename
        while True:
            temp_link_name = tempfile.mktemp(dir=link_dir)

            # os.* functions mimic as closely as possible system functions
            # The POSIX symlink() returns EEXIST if link_name already exists
            # https://pubs.opengroup.org/onlinepubs/9699919799/functions/
            # symlink.html
            try:
                os.symlink(target, temp_link_name)
                break
            except FileExistsError:
                pass

        # Replace link_name with temp_link_name
        try:
            # Pre-empt os.replace on a directory with a nicer message
            if not os.path.islink(link_name) and os.path.isdir(link_name):
                raise IsADirectoryError(
                    f"Cannot symlink over existing directory: '{link_name}'")
            os.replace(temp_link_name, link_name)
        except:
            if os.path.islink(temp_link_name):
                os.remove(temp_link_name)
            raise

    #def variablename(self,var):
    #    return [tpl[0] for tpl in filter(
    #        lambda x: var is x[1], globals().items())]

    def flatten2list(self,object):
        gather = []
        for item in object:
            if isinstance(item, (tuple, set)):
                gather.extend(self.flatten2list(item))            
            else:
                gather.append(item)
        return gather

    def prepare_lc(self, tpf, add_bkg):
        """
        For a given target, it returns the light curve masked as needed 
        and the set of valid background pixels.
        
        bkg : boolean variable that when true, the background flux is added 
        back in step 0 below
        
        It proceeds through the following steps:

            0) If chosen so, adds the background flux to the tpf
                Note: The background flux errors are correspondingly 
                subtracted in quadrature
            1) Obtains the lc via simple aperture photometry (sap)
            2) If selected to do so, removes outliers 
            3) If needed, creates and applies a transit mask
            4) Creates and applies a nan mask
            5) Parses the background aperture mask and obtains the background 
               pixel fluxes. 
                   Note: Flux not normalized background components for TESS by 
                   default #TODO: CHECK
            6) Removes nans from the background pixel fluxes
        """
        
        if add_bkg == True:
            tpf += tpf.flux_bkg
            
            #Procedure to include the background flux errors using the existing
            #add and multiplication mechanisms of the tpf class
            tpf_flux = tpf.flux
            #Ratio of the flux errors with the background flux errors subtracted 
            #in quadrature and the original flux errors
            mult_corr = np.sqrt( (tpf.flux_err)**2 - \
                                (tpf.flux_bkg_err)**2 ) / tpf.flux_err
            #Including the new flux errors using the multiplication mechanism
            tpf *= mult_corr
            #Correcting the flux by adding the intended flux+background-flux 
            #and subtracting the flux which was multiplied by the ratio of 
            #errors
            tpf += tpf_flux * (1 - mult_corr)

        #TODO: The default for the 'sap' method is an aperture_mask of 
        #'default', which corresponds to 'pipeline' if exists or 'threshold' 
        #otherwise. This should be changed consistently if/when allowing 
        #'aperture_mask' to be changed as a parameter in PLD
        lc = tpf.to_lightcurve(method='sap')
        
        #TODO: This test method could be implemented as a correction 
        #for 'to_lightcurve'
        #test method: converts lc to lc_with_spoc_sap_flux_err
        #lc_with_spoc_sap_flux_err = self.to_spoc_sap_flux_err(lc,tpf)
        
        if self.remove_outliers==True:
            lc.remove_outliers()
        
        #TODO: Consider using a transit mask
        #transit_mask = lc.create_transit_mask(
        #    transit_time=transit_time, period=period, duration=duration)
        #lc  = lc[~transit_mask]
        #tpf = tpf[~transit_mask]

        nan_mask = np.isnan(lc.flux)
        tpf = tpf[~nan_mask]
        lc  = lc[~nan_mask]
        
        if add_bkg == True:
            background_aperture_mask = tpf._parse_aperture_mask('background')
            bkg_pixels = tpf.flux[:, background_aperture_mask].reshape(
                len(tpf.flux), -1)
            bkg_pixels = bkg_pixels.value
            bkg_pixels = np.array([r[np.isfinite(r)] for r in bkg_pixels])
            return tpf, lc, bkg_pixels
        
        return tpf, lc

    def to_spoc_sap_flux_err(self,lc,tpf):
        """
        Test method that converts the flux errors of an lc from those obtained 
        from the 'to_lightcurve' method with option 'sap' of the 'tpf' class of 
        Lightkurve to the flux errors of the SPOC SAP light curve. The former 
        does not account for pixel correlations of the background flux errors, 
        while the latter accounts for a full correlation.
        """
        
        lc_to_convert = copy.deepcopy(lc)
        
        #TODO: The selection of the pixel could be generalized in case the 
        #one at 0,0 has not valid background flux errors
        lc_to_convert.flux_bkg_err = tpf.flux_bkg_err[:,0,0]
                    
        #TODO: The calculation of the number of pixels could be generalized to
        #allow other masks using the method 'self.npix'
        npix = tpf.pipeline_mask.sum()  
      
        lc_to_convert.flux_err = np.sqrt( (lc_to_convert.flux_err)**2 \
                                        -( np.sqrt(npix) * \
                                          lc_to_convert.flux_bkg_err)**2 \
                                        + (npix*lc_to_convert.flux_bkg_err)**2 )    
        return lc_to_convert
    
    def npix(self,tpf):
        """
        Calculates the number of pixels for the given aperture_mask in PLD
        """

        if self.parameters['PLD']['aperture_mask']=='pipeline':
            npix = tpf.pipeline_mask.sum()
        #TODO: To allow 'aperture_mask' to change, consistency must be 
        #ensured in the 'to_lightcurve' method in 'self.prepare_lc'
        else:
            raise ValueError('Only the pipeline mask is currently available')
        #elif self.parameters['PLD']['aperture_mask']=='threshold':
        #    npix = tpf.create_threshold_mask().sum()
        
        return npix
    
    def RCQ(self, cindex, tpf, lc, bkg_pixels, quat_file):
        """
        a) Runs the RegressionCorrector followign H21.
            1) Sets Gaussian prior
            2) Creates background design matrix
            3) Applies PCA and Gaussian prior
            4) Implements spline design matrix to capture stellar variability
            
        b) Restores the spline to preserve stellar variability
        """

        corr='RCQ'
        
        paramvar_list, param_idxs = self.paramvar_struct(corr)

        for pidx, paramvar in enumerate(paramvar_list):
            prior_sigma = np.nanstd(tpf.flux.value) * ntimes_nanstd
            bkg_dm = DesignMatrix(bkg_pixels, name="Background")
            bkg_dm = bkg_dm.pca(paramvar[param_idxs['pca_components']])
            bkg_dm.prior_sigma = np.ones(bkg_dm.shape[1]) * prior_sigma
            filtered_quaternion_dm = filtered_quat_dm(lc,quat_file)
        
            spline_dm = create_spline_matrix(
                np.arange(lc.flux.size), 
                n_knots=paramvar[param_idxs['spline_n_knots']], name='Spline')
            dmc = DesignMatrixCollection([bkg_dm, filtered_quaternion_dm, 
                                          spline_dm])
            regressionc_lc = RegressionCorrector(lc)
            corrected_lc = regressionc_lc.correct(
                dmc, propagate_errors=self.propagate_errors)
            
            spline_lc = regressionc_lc.diagnostic_lightcurves['Spline']
            median_spline_lc = np.median(spline_lc.flux)
            corrected_lc = corrected_lc + spline_lc - median_spline_lc
            
            if self.corr_diags==True:
                regressionc_lc.diagnose()
                
            corrected_lc=self.flux_level_adjust(tpf,corrected_lc,bkg_pixels)
            corrected_lc=self.flux_adjust(tpf,corrected_lc)
            
            if pidx==0:
                full_corr_lc = copy.deepcopy(corrected_lc)
            full_corr_lc.add_column(
                corrected_lc.flux,
                name='{}_flx_parvar_{}'.format(corr,pidx))
            full_corr_lc.add_column(
                corrected_lc.flux_err,
                name='{}_flx_err_parvar_{}'.format(corr,pidx))
        
        return full_corr_lc

    def PLD(self, cindex, tpf, bkg_pixels):
        """
        Runs the PLDCorrector for the required corrector parameter variations
        """
        
        corr='PLD'
        
        paramvar_list, param_idxs = self.paramvar_struct(corr)
                
        pld = PLDCorrector(tpf, aperture_mask=aperture_mask)
        
        for pidx, paramvar in enumerate(paramvar_list):
            paramvar_dict={}
            for param_name in self.param_names[corr]:
                paramvar_dict[param_name] = paramvar[param_idxs[param_name]]
            corrected_lc = pld.correct(restore_trend=True,**paramvar_dict,
                                       propagate_errors=self.propagate_errors)
            
            if self.corr_diags==True:
                pld.diagnose()
                pld.diagnose_masks()
                
            corrected_lc=self.flux_level_adjust(tpf,corrected_lc,bkg_pixels)
            corrected_lc=self.flux_adjust(tpf,corrected_lc)
            
            if pidx==0:
                full_corr_lc = copy.deepcopy(corrected_lc)
            full_corr_lc.add_column(
                corrected_lc.flux,
                name='{}_flx_parvar_{}'.format(corr,pidx))
            full_corr_lc.add_column(
                corrected_lc.flux_err,
                name='{}_flx_err_parvar_{}'.format(corr,pidx))
        
        return full_corr_lc
        
    def CBV(self,cindex,cadence_mask,tpf,lc):
        """
        Runs the CBVCorrector plus a spline that gets restored.
        
        spline_flag  : allows to remove the use of the spline
        auto_reg     : switches off automatic regularization
        alpha_reg    : value of the fixed regularization when the automatic
                       one is switched off
        """
        
        corr='CBV'
        
        paramvar_list, param_idxs = self.paramvar_struct(corr)
        
        lc_masked = copy.deepcopy(lc[cadence_mask])
        cbvCorrector = CBVCorrector(lc_masked,cbv_dir=self.cbv_dir)
        
        for pidx, paramvar in enumerate(paramvar_list): 
            if paramvar[param_idxs['spline_flag']] == True:
                spline_dm = create_spline_matrix(
                    np.arange(lc.flux.size),
                    n_knots=paramvar[param_idxs['spline_n_knots']],
                    name='Spline')                
                masking_spline = spline_dm.values[cadence_mask]
                spline_masked_dm = DesignMatrix(masking_spline,
                                                name='Spline_masked')
                ext_dm=spline_masked_dm
            else:
                ext_dm=None
            
            cbv_type_entry=paramvar[param_idxs['cbv_type']]
            cbv_indices_entry=paramvar[param_idxs['cbv_indices']]
            
            if auto_reg == True:
                corrected_lc=cbvCorrector.correct(
                    cbv_type=cbv_type_entry,cbv_indices=cbv_indices_entry,
                    ext_dm=ext_dm,cadence_mask=None,
                    alpha_bounds=[0.0001, 10000.0],target_over_score=0.8,
                    target_under_score=0.8,max_iter=100)
                paramvar[param_idxs['alpha_reg']]=cbvCorrector.alpha
            else:
                corrected_lc=cbvCorrector.correct_gaussian_prior(
                    cbv_type=cbv_type_entry,cbv_indices=cbv_indices_entry,
                    ext_dm=ext_dm,cadence_mask=None,
                    alpha=paramvar[param_idxs['alpha_reg']])
            
            if paramvar[param_idxs['spline_flag']] == True:
                #restoring spline to preserve stellar variability
                spline_lc = cbvCorrector.diagnostic_lightcurves['Spline_masked']
                median_spline_lc = np.median(spline_lc.flux)
                corrected_lc = corrected_lc + spline_lc - median_spline_lc
            
            if self.corr_diags==True:
                cbvCorrector.diagnose()
            
            corrected_lc=self.flux_adjust(tpf,corrected_lc)
            
            if pidx==0:
                full_corr_lc = copy.deepcopy(corrected_lc)
            full_corr_lc.add_column(
                corrected_lc.flux,
                name='{}_flx_parvar_{}'.format(corr,pidx))
            full_corr_lc.add_column(
                corrected_lc.flux_err,
                name='{}_flx_err_parvar_{}'.format(corr,pidx))
            
        return full_corr_lc
    
    def paramvar_struct(self,corr):
        """
        Prepares the data structure for saving corrector parameter variations
        """
        
        params = list(self.parameters[corr].keys())
        for param in params:
            globals()[param] = self.parameters[corr][param]
        
        params_dict = self.parameters[corr]['params_to_var']
        param_names = list(params_dict.keys())                
        param_values = []
        param_idxs = {}
        for param_idx, param_name in enumerate(param_names):
            globals()[param_name] = params_dict[param_name]
            param_value = globals()[param_name]
            param_idxs[param_name]=param_idx
            param_values.append(param_value)
            
        if corr=='CBV':
            #Linking cbv_type and cbv_indices
            cbv_item = list(zip(cbv_type,cbv_indices))
            #updating the param_values list
            param_values.remove(cbv_type)
            param_values.remove(cbv_indices)
            param_values.append(cbv_item)
            #updating the param_names list
            param_names.remove('cbv_type')
            param_names.remove('cbv_indices')
            param_names.append('cbv_type')
            param_names.append('cbv_indices')
            #Linking spline_flag,spline_n_knots
            spline_item = list(zip(spline_flag,spline_n_knots))
            #updating the param_values list
            param_values.remove(spline_flag)
            param_values.remove(spline_n_knots)
            param_values.append(spline_item)
            #updating the param_names list
            param_names.remove('spline_flag')
            param_names.remove('spline_n_knots')
            param_names.append('spline_flag')
            param_names.append('spline_n_knots')
            
            #updating the param_idxs dictionary
            param_idxs={}
            for param_idx, param_name in enumerate(param_names):
                param_idxs[param_name]=param_idx
        
        lenpv=len(param_values)
        paramvar=param_values[0]
        if lenpv>1:
            for param_value in param_values[1:]:
                paramvar=list(product(paramvar,param_value))
            
        paramvar_list=[]
        for pitem in paramvar:
            fpitem=np.array(self.flatten2list(pitem),dtype=object)
            paramvar_list.append(fpitem) 
        paramvar_arr=np.array(paramvar_list)

        self.param_names[corr] = param_names
        self.paramvar_arr[corr] = paramvar_arr
        
        return paramvar_list, param_idxs
        
    def diagnose_paramvar(self,corr,cindex,corr_lc,
                          save_name,overwrite_ok,path=None):
        """
        Plots a diagnostic figure for the lightcurves of a set of parameter
        values for each varying parameter of a corrector
        """
        
        # if self.diag_parName==None:
        #     print('For parameter variation diag_pvar, recall to input a '\
        #           'parameter name in diag_parName.')
        #     return
        
        tic    = self.cases_list[cindex]['TIC']
        sector = self.cases_list[cindex]['Sector']        
        case   = self.cases_list[cindex]['Case_Index']
        
        title = 'TIC {} Sector {} Case {}'.format(tic, sector, case)
        
        #if from_comp_plt==True:
        if path==None:
            path=corr_lc.meta['FILENAME']
        with fits.open(path) as hdu:
            self.paramvar_arr[corr] = hdu['{} PARAMVAR'.format(corr)].data
            self.param_names[corr] = hdu['{} PARAMVAR'.format(
                corr)].columns.names
        
        paramvar_list = list(self.paramvar_arr[corr])
        plength = len(paramvar_list)
        param_names = self.param_names[corr]
        
        subtitle = '{}'.format(
            param_names).strip('[]').replace("\'","").replace(',',' - ')
        
        alpha=0.8
        colormap = cm.tab20c
        colors = iter(colormap(np.linspace(1, 0, plength)))
        # colormap = cm.tab20b
        # colors = iter(colormap(np.linspace(0, 1, plength)))
        
        handles=[]
        labels=[]
        
        fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(8,7))
        
        columns = [x for x in corr_lc.colnames if 'flx_parvar' in x]
        columns[0] = 'flux'
        for paramvar,column in reversed(list(zip(paramvar_list,columns))):
            #to make cbv indices arrays more compact for the legend
            if corr == 'CBV':
                cbv_indices_idx = param_names.index('cbv_indices')
                pvarList = paramvar[cbv_indices_idx].strip('[]').replace(
                    '), ','').strip('array(').replace(
                    "\'ALL\'","array(ALL").split('array(')
                paramvar[cbv_indices_idx] = str(['({}-{})'.format(
                    x.strip('[]').split(', ')[0],x.strip(
                        '[]').split(', ')[-1]) if x!='ALL' else x \
                                   for x in pvarList])
            #formatting the label
            label = str(paramvar).strip(
                '[]').replace("\'","").replace('\"','').replace(
                "list","").replace("array","").replace(
                "(","").replace(")","").replace('MultiScale','MultiSc').replace(
                'SingleScale','SingleSc')
            color=next(colors)
            corr_lc.scatter(
                column=column,label=label,alpha=alpha,color=color,ax=ax)
        
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        
        ax.legend(by_label.values(),by_label.keys(),
                  bbox_to_anchor=(0., 1.02, 1.0, 0.2),
                  ncol=1, mode="expand", borderaxespad=0, borderpad=2,
                  markerscale=5,scatterpoints=3)
        ax.get_legend().set_title(
            '{} Parameter Variations for {} \n\n{}'.format(
                corr,title,subtitle))
        
        path_to_pkl_file = '{}{}/{}.pkl'.format(
            dir_to_files,save_name,save_name)
        path_to_pkl_file, path_to_dir = self.set_file_dir(
            path_to_pkl_file,save_name,overwrite_ok)
        path_to_png=path_to_dir+'png_files'
        makedirs(path_to_png,exist_ok=True)
        
        file='{}/paramvar_fig_corrector{}-tic-{}-sector-{}-'\
        'cindex-{}.png'.format(path_to_png,corr,tic,sector,case)
        fig.savefig(file, bbox_inches="tight", facecolor=(1, 1, 1))   
        
        return
    
    def flux_adjust(self,tpf,corrected_lc):
        """
        Adjusts:
            1) flux fraction in aperture 
            2) crowding
        """
        
        flfrcsap = tpf.get_header(ext=1).get('FLFRCSAP')
        crowdsap = tpf.get_header(ext=1).get('CROWDSAP')

        corrected_lc.flux = (corrected_lc.flux - \
                             (1-crowdsap)*np.median(corrected_lc.flux)
                            ) / flfrcsap
        #Flux fraction scaling of the flux errors; crowding scaling is 
        #considered negligible as in the SPOC pipeline
        corrected_lc.flux_err = corrected_lc.flux_err / flfrcsap
        
        return corrected_lc

    def flux_level_adjust(self,tpf,corrected_lc,bkg_pixels):
        """
        Adjusts the flux level, in case the mean of the uncorrected flux 
        was preserved in a regression.
        """
        
        bkg_pixels = Quantity(bkg_pixels, corrected_lc.flux.unit)
            
        if self.parameters['RCQ']['bkg_adjust']=='mean' or \
        self.parameters['PLD']['bkg_adjust']=='mean':
            bkg = np.nanmean(bkg_pixels, axis=0)
        else:
            bkg = np.nanmedian(bkg_pixels, axis=0)
            
        bkg0 = np.sort(bkg)[2]        #3rd dimmest median pixel
        #bkg0 = np.sort(bkg)[10]         #X dimmest median pixel
        #bkg0 = np.percentile(bkg, 5) #5 percentile
        #TODO: CHECK USING FULL APERTURE
            
        npix=self.npix(tpf)
        corrected_lc.flux = corrected_lc.flux - npix*bkg0
            
        return corrected_lc
    
    def comparison_plots(self,cindex,correctors=None,
                         msize=0.3,alpha=0.3,norm=False,SAP_flag=True,
                         save_name=None,overwrite_ok=False,
                         plot_type='joined_panels'):
        """
        For a given set of correctors, it plots a figure for each 
        comparing the corrected light curve to the PDCSAP and SAP (optional) 
        light curves. This can be performed for a target case or all of those 
        in the target list.
    
        cindex    : index of a given case or 'all' to plot all cases in 
                    the current bin
        msize     : matplotlib size of the marker
        alpha     : matplotlib transparency parameter
        norm      : boolean to select normalize light curves or not
        SAP_flag  : boolean to select the display of the SAP light curve
        save_name : it sets the name of the directory storing png files
                    of light curve comparisons. The same string is used as
                    the start of the filename of these files
        plot_type : select between these types of plots to show the results 
                    from the different correctors:
                        1) joined_panels (default)
                        2) separate_plots
                        3) single_panel
        """

        self.load_saved_input(save_name=save_name)
        
        if correctors==None:
            correctors=self.correctors
        else:
            self.valid_corrs(correctors)
        
        if cindex == 'all':
            for cidx in np.arange(self.first_cindex+1,len(self.cases_list)+1):
                self.comparison_plot(cidx,correctors,msize,alpha,
                                     norm,SAP_flag,save_name,overwrite_ok,
                                     plot_type)
                with plt.ioff():
                    plt.close()
        else:
            self.comparison_plot(cindex,correctors,msize,alpha,
                                 norm,SAP_flag,save_name,overwrite_ok,plot_type)
        
        return
        
    def comparison_plot(self,cindex,correctors,msize,alpha,norm,SAP_flag,
                        save_name,overwrite_ok,plot_type):
        """
        For a given set of correctors, it plots a figure for each 
        comparing the corrected light curve to the PDCSAP and SAP (optional) 
        light curves. This is performed for a given case in the target list.
    
        Same parameters as the <comparison_plots> method.
        """
        
        self.load_saved_metrics(save_name=save_name)

        cindex = cindex-1
        
        tic    = self.cases_list[cindex]['TIC']
        sector = self.cases_list[cindex]['Sector']        
        case   = self.cases_list[cindex]['Case_Index']
        
        if save_name == None:
            save_name=input('Enter name of saving directory: ')
            
        path_to_pkl_file = '{}{}/{}.pkl'.format(
            dir_to_files,save_name,save_name)
        path_to_pkl_file, path_to_dir = self.set_file_dir(
            path_to_pkl_file,save_name,overwrite_ok)
        path_to_png=path_to_dir+'png_files'
        makedirs(path_to_png,exist_ok=True)

        spoc_lc = lk.read('{}'.format(self.spoc_lc_filenames[cindex]),
                          quality_bitmask=quality_bitmask)
        
        #TODO: MOVE LISTS TO PLOTTING
        path_to_files = dir_to_files+save_name+'/'
        path_to_sap_files = path_to_files+'fits_files/sap/'
        filenames_sap = next(walk(path_to_sap_files), (None, None, []))[2]
        for filename in filenames_sap:
            cidx=(filename.removesuffix('.fits').split('-')[-1])
            if int(cidx)==int(case):
                sap_lc = lk.read('{}{}'.format(path_to_sap_files,filename),
                                quality_bitmask=quality_bitmask)
                
        path_to_corrected_files = path_to_files+'fits_files/corrected/'
        filenames_corrected = \
        next(walk(path_to_corrected_files), (None, None, []))[2]
        corrected_lc={}
        for filename in filenames_corrected:
            cidx=(filename.removesuffix('.fits').split('-')[-1])
            if int(cidx)==int(case):
                corrector=(filename.removesuffix('.fits').split('-')[1])
                corrected_lc[corrector] = lk.read('{}{}'.format(
                    path_to_corrected_files,
                    filename), quality_bitmask=quality_bitmask)
        
        if self.diag_pvar==True:
            for corrector in correctors:
                self.diagnose_paramvar(corrector,cindex,
                                       corrected_lc[corrector],
                                       save_name,overwrite_ok)
        
        if plot_type=='separate_plots':
            for corrector in correctors:
                #TODO: TESTING FILENAMES
                #ax = self.spoc_lc_list[cindex].scatter(
                ax = spoc_lc.scatter(
                    c='red', s=msize, alpha=alpha, label='PDCSAP',
                    normalize=norm)
                if SAP_flag == True:
                    #TODO: TESTING FILENAMES
                    #ax = self.sap_lc_list[cindex].scatter(
                    ax = sap_lc.scatter(
                        ax=ax, c='orange', s=msize, alpha=alpha, label='SAP',
                        normalize=norm)
                #TODO: TESTING FILENAMES
                #ax = self.corrected_lc_list[cindex][corrector].scatter(
                ax = corrected_lc[corrector].scatter(
                    ax=ax, c=self.parameters[corrector]['color'], s=msize, 
                    alpha=alpha, label='{}'.format(corrector), 
                    normalize=norm).legend(
                    title='TIC {} Sector {}'.format(tic, sector),
                    markerscale=5,scatterpoints=3)
                #if save_name != None:
                file='{}/fig_corrector-{}-tic-{}-sector-{}-'\
                'cindex-{}.png'.format(path_to_png,corrector,tic,sector,case)
                ax.figure.savefig(file, facecolor=(1, 1, 1))

        if plot_type=='single_panel':
            #TODO: TESTING FILENAMES
            #ax = self.spoc_lc_list[cindex].scatter(
            ax = spoc_lc.scatter(
                c='red', s=msize, alpha=alpha, label='PDCSAP',
                normalize=norm)
            if SAP_flag == True:
                #TODO: TESTING FILENAMES
                #ax = self.sap_lc_list[cindex].scatter(
                ax = sap_lc.scatter(
                    ax=ax, c='orange', s=msize, alpha=alpha, label='SAP',
                    normalize=norm)
            for corrector in correctors:
                #TODO: TESTING FILENAMES
                #ax = self.corrected_lc_list[cindex][corrector].scatter(
                ax = corrected_lc[corrector].scatter(
                    ax=ax, c=self.parameters[corrector]['color'], s=msize, 
                    alpha=alpha, label='{}'.format(corrector), 
                    normalize=norm)
            ax.legend(title='TIC {} Sector {}'.format(tic, sector),
                      markerscale=5,scatterpoints=3)
            #if save_name != None:
            combined_correctors=''
            for corrector in correctors:
                combined_correctors+='-'+corrector
            file='{}/fig_one_panel_correctors{}-tic-{}-sector-{}-'\
            'cindex-{}.png'.format(
                path_to_png,combined_correctors,tic,sector,case)
            ax.figure.savefig(file, facecolor=(1, 1, 1))
                    
        if plot_type=='joined_panels':
            num_subplots=len(correctors)+1
            ax = np.empty(num_subplots, dtype=object)
            if num_subplots>1:
                fig, ax = plt.subplots(num_subplots,1,sharex=True,
                                      figsize=(7,7))
                fig.subplots_adjust(hspace=0,top=0.75)
            else:
                fig, ax[0] = plt.subplots(num_subplots,1,sharex=True)
            fig.suptitle('TIC {} Sector {}'.format(tic, sector),y=0.98)
            
            isub=0
            handles=[]
            labels=[]
            
            if SAP_flag == True:
                SAP_CDPP=[]
                for minutes in minutes_list:
                    SAP_CDPP.append(
                        self.metrics_list[cindex][correctors[0]]\
                    ['sap_sgCDPP_{}min'.format(minutes)]*\
                        u.dimensionless_unscaled.to(cds.ppm)*cds.ppm)
                #TODO: TESTING FILENAMES    
                #self.sap_lc_list[cindex].scatter(
                sap_lc.scatter(
                    ax=ax[isub],c='orange',s=msize,alpha=alpha,
                    label='SAP   1.00   {0[0]:.0f}   '\
                    '{0[1]:.0f}   {0[2]:.0f}   {0[3]:.0f}   '\
                    '{0[4]:.0f}'.format(SAP_CDPP), normalize=norm)
            PDC_CDPP=[]
            for minutes in minutes_list:
                    PDC_CDPP.append(
                        self.metrics_list[cindex][correctors[0]]\
                    ['pdc_sgCDPP_{}min'.format(minutes)]*\
                        u.dimensionless_unscaled.to(cds.ppm)*cds.ppm)
            POS=self.metrics_list[cindex][correctors[0]]['pdc_overfit']
            #TODO: TESTING FILENAMES
            #self.spoc_lc_list[cindex].scatter(
            axLine, axLabel = \
            spoc_lc.scatter(
                ax=ax[isub], c='red', s=msize, alpha=alpha,
                label='PDC   {0:.2f}   {1[0]:.0f}   {1[1]:.0f}   '\
                '{1[2]:.0f}   {1[3]:.0f}   {1[4]:.0f}'.format(POS,PDC_CDPP),
                normalize=norm).get_legend_handles_labels()
            ax[isub].tick_params(right=True,direction='in')
            handles.extend(axLine)
            labels.extend(axLabel)
            isub+=1
            
            for corrector in correctors:
                CDPP=[]
                for minutes in minutes_list:
                    CDPP.append(self.metrics_list[cindex][corrector]\
                    ['sgCDPP_{}min'.format(minutes)]*\
                        u.dimensionless_unscaled.to(cds.ppm)*cds.ppm)
                
                OS=self.metrics_list[cindex][corrector]['Over_fit']
                
                if SAP_flag == True:
                    #TODO: TESTING FILENAMES    
                    #self.sap_lc_list[cindex].scatter(
                    sap_lc.scatter(
                        ax=ax[isub],c='orange',s=msize,alpha=alpha,
                        label='SAP   1.00   {0[0]:.0f}   '\
                        '{0[1]:.0f}   {0[2]:.0f}   {0[3]:.0f}   '\
                        '{0[4]:.0f}'.format(SAP_CDPP),
                        normalize=norm).yaxis.set_tick_params(labeltop=True)
                #TODO: TESTING FILENAMES
                #self.corrected_lc_list[cindex][corrector].scatter(
                axLine, axLabel = \
                corrected_lc[corrector].scatter(
                    ax=ax[isub], c=self.parameters[corrector]['color'], 
                    s=msize, alpha=alpha,
                    label='{0}   {1:.2f}   {2[0]:.0f}   {2[1]:.0f}   '\
                    '{2[2]:.0f}   {2[3]:.0f}   {2[4]:.0f}'.format(
                        corrector,OS,CDPP),
                    normalize=norm).get_legend_handles_labels()
                ax[isub].get_legend().remove()
                ax[isub].tick_params(right=True,direction='in',
                                     labelright=False)
                handles.extend(axLine)
                labels.extend(axLabel)
                by_label = dict(zip(labels, handles))
                isub+=1
                
            ax[0].legend(
                by_label.values(), by_label.keys(), 
                bbox_to_anchor=(0., 1.02, 1.0, .102), loc='lower left',
                ncol=1, mode="expand", borderaxespad=0.,
                markerscale=10, scatterpoints=3,
                title='Method - Overfit Score - '\
                'sgCDPP ({}min, {}min, {}min, {}min, {}min)'.format(
                    *minutes_list))
        
            plt.show()
        
            #if save_name != None:
            combined_correctors=''
            for corrector in correctors:
                combined_correctors+='-'+corrector
            file='{}/fig_combined_correctors{}-tic-{}-sector-{}-'\
            'cindex-{}.png'.format(
                path_to_png,combined_correctors,tic,sector,case)
            fig.savefig(file, facecolor=(1, 1, 1))
        
        return
    
    #TODO: Move metrics calculations to a new class Metrics_Provider
    def metrics_calculate(self, corrected_lc, lc, spoc_lc, sap_lc):
        """
        Obtains the following metrics:
        
        'sgCDPP_<>min' : sgCDPP (Savitzky-Golay, Combined Differential 
                         Photometric Precision) via the “sgCDPP proxy 
                         algorithm” implemented in Lightkurve [see Gilliland
                         et al. (2011) and Van Cleve et al. (2016)], in ppm
                         (parts per million). For various transit 
                         durations (in minutes).
            
         'pdc_sgCDPP_<>min' : The sgCDPPs for the PDCSAP light curve
         'sap_sgCDPP_<>min' : The sgCDPPs for the SAP light curve 
         
         'pdc_CDPP_1h' : The CDPP calculated by the SPOC pipeline using a 
                         wavelet based method algorithm to calculate the 
                         signal-to-noise ratio of the specific waveform of
                         transits of various durations [see Christiansen 
                         et al. (2012)]. For 1 hour transit duration in ppm.
         'Over_fit'  : Lomb-Scargle periodogram metric for introduced noise
         'Under_fit' : Metric for residual correlations with neighbor targets 
                       (Currently not used since it would require adaptations 
                       to fairly compare with neighbours; note also that it 
                       requires connectivity during the run and that it appears
                       to slowdown the run significantly) #TODO: CHECK
         'pdc_overfit'  : PDCSAP Lomb-Scargle metric
         'spoc_pdc_overfit'  : SPOC PDCSAP Lomb-Scargle metric
         'pdc_underfit' : SPOC PDCSAP residual correlations metric
        """
        
        metrics={}
        
        ind=0
        for transit_duration in transit_durations:
            sgcdpp_corr=float(corrected_lc.estimate_cdpp(
                transit_duration=int(transit_duration), 
                savgol_window=int(savgol_windows[ind]),
                savgol_polyorder=savgol_polyorder))
            metrics['sgCDPP_{}min'.format(
                minutes_list[ind])]=sgcdpp_corr
            
            sgcdpp_pdc=float(spoc_lc.estimate_cdpp(
                transit_duration=int(transit_duration), 
                savgol_window=int(savgol_windows[ind]),
                savgol_polyorder=savgol_polyorder).unmasked)
            metrics['pdc_sgCDPP_{}min'.format(
                minutes_list[ind])]=sgcdpp_pdc
            
            sgcdpp_sap=float(sap_lc.estimate_cdpp(
                transit_duration=int(transit_duration), 
                savgol_window=int(savgol_windows[ind]),
                #the unmasked quantity might be needed if using the SPOC SAP lc
                #but it fails when using the SAP lc from to_lightcurve
                #savgol_polyorder=savgol_polyorder).unmasked)
                savgol_polyorder=savgol_polyorder))
            metrics['sap_sgCDPP_{}min'.format(
                minutes_list[ind])]=sgcdpp_sap
            
            ind+=1
        
        metrics['pdc_CDPP_1h'] = (spoc_lc.meta['CDPP1_0'])
        
        metrics['Over_fit'] = overfit_metric_lombscargle(sap_lc,corrected_lc,
                                                         n_samples=n_samples)
        #For the calculation of the PDC overfit, use the sap lc from SPOC to
        #have consistent errors (i.e., accounting for full correlation between 
        #the background errors)
        spoc_sap_lc = copy.deepcopy(spoc_lc)
        spoc_sap_lc.flux = spoc_sap_lc['sap_flux']
        spoc_sap_lc.flux_err = spoc_sap_lc['sap_flux_err']
        metrics['pdc_overfit'] = overfit_metric_lombscargle(spoc_sap_lc,spoc_lc,
                                                            n_samples=n_samples)
        #metrics['Under_fit'] = underfit_metric_neighbors(corrected_lc)
        metrics['spoc_pdc_overfit'] = spoc_lc.meta['PDC_NOI']
        #metrics['pdc_underfit'] = spoc_lc.meta['PDC_COR']
        
        return metrics

    #TODO: new version of the metrics_calculate method to be incoporated 
    #into the rest of metrics methods
    def new_metrics_calculate(self, corrected_lc, sap_lc):
        """
        Obtains the following metrics:
        
        'sgCDPP_<>min' : sgCDPP (Savitzky-Golay, Combined Differential 
                         Photometric Precision) via the “sgCDPP proxy 
                         algorithm” implemented in Lightkurve [see Gilliland
                         et al. (2011) and Van Cleve et al. (2016)], in ppm
                         (parts per million). For various transit 
                         durations (in minutes).
         'Over_fit'  : Lomb-Scargle periodogram metric for introduced noise
         'Under_fit' : Metric for residual correlations with neighbor targets 
                       (Currently not used since it would require adaptations 
                       to fairly compare with neighbours; note also that it 
                       requires connectivity during the run and that it appears
                       to slowdown the run significantly) #TODO: CHECK
        """
        
        metrics={}
        
        ind=0
        for transit_duration in transit_durations:
            sgcdpp_corr=float(corrected_lc.estimate_cdpp(
                transit_duration=int(transit_duration), 
                savgol_window=int(savgol_windows[ind]),
                savgol_polyorder=savgol_polyorder))
            metrics['sgCDPP_{}min'.format(
                minutes_list[ind])]=sgcdpp_corr
            
            ind+=1
        
        metrics['Over_fit'] = overfit_metric_lombscargle(sap_lc,corrected_lc,
                                                         n_samples=n_samples)
        
        return metrics
    
    def metric_statistics(self,save_name,overwrite_ok,correctors=None,
                          output='print',cindex=1,
                          bins=30,alpha=0.5,density=False,spoc_pdc=False,
                          cumulative=False,histtype='stepfilled',log=False,
                          weights_combined_score=None):
        """
        Loads json metric files into a metrics_list
        
        Prints and plots metric statistics
        
        Parameter:
                output: 'print' (default): prints metrics for a given case
                        'overfit': plots overfitting scores
                        'sgcdpp' : plots sgCDPP results
                        'changes': adds or recalculates metrics
        """

        self.load_saved_input(save_name=save_name)
        self.load_saved_metrics(save_name=save_name)
        
        if output=='print':
            if correctors==None:
                correctors=self.correctors
            else:
                self.valid_corrs(correctors)
        
            for corrector in correctors:
                ind=0
                for transit_duration in transit_durations:
                    key='sgCDPP_{}min'.format(minutes_list[ind])
                    sgcdpp=self.metrics_list[cindex][corrector][key]*\
                    u.dimensionless_unscaled
                    sgcdpp=sgcdpp.to(cds.ppm)
                    print("{}: sgCDPP ({} minutes) = {:.0f}"\
                          .format(corrector,minutes_list[ind],sgcdpp))
                    ind+=1
                key='sgCDPP_harmonic_mean'
                sgcdpp=self.metrics_list[cindex][corrector][key]*\
                u.dimensionless_unscaled
                sgcdpp=sgcdpp.to(cds.ppm)
                print("{}: sgCDPP harmonic mean = {:.0f}"\
                      .format(corrector,sgcdpp))
                print("\n")
            
            ind=0
            for transit_duration in transit_durations:
                key='pdc_sgCDPP_{}min'.format(minutes_list[ind])
                sgcdpp=self.metrics_list[cindex][correctors[0]][key]*\
                u.dimensionless_unscaled
                sgcdpp=sgcdpp.to(cds.ppm)
                print("PDC: sgCDPP ({} minutes) = {:.0f}"\
                      .format(minutes_list[ind],sgcdpp))
                ind+=1
            key='pdc_sgCDPP_harmonic_mean'
            sgcdpp=self.metrics_list[cindex][correctors[0]][key]*\
            u.dimensionless_unscaled
            sgcdpp=sgcdpp.to(cds.ppm)
            print("PDC: sgCDPP harmonic mean = {:.0f}"\
                  .format(sgcdpp))
            
            print("\n")

            cdpp=self.metrics_list[cindex][correctors[0]]['pdc_CDPP_1h']
            print("PDC SPOC: CDPP (1h) = {:.0f}".format(cdpp))

            print("\n")
            
            ind=0
            for transit_duration in transit_durations:
                key='sap_sgCDPP_{}min'.format(minutes_list[ind])
                sgcdpp=self.metrics_list[cindex][correctors[0]][key]*\
                u.dimensionless_unscaled
                sgcdpp=sgcdpp.to(cds.ppm)
                print("SAP: sgCDPP ({} minutes) = {:.0f}"\
                      .format(minutes_list[ind],sgcdpp))
                ind+=1
            key='sap_sgCDPP_harmonic_mean'
            sgcdpp=self.metrics_list[cindex][correctors[0]][key]*\
            u.dimensionless_unscaled
            sgcdpp=sgcdpp.to(cds.ppm)
            print("SAP: sgCDPP harmonic mean = {:.0f}"\
                  .format(sgcdpp))
        
            print("\n")
    
            for corrector in correctors:
                print("{}: Overfitting score  = {:.2f}"\
                      .format(corrector,
                              self.metrics_list[cindex][corrector]['Over_fit']))
                #print("{}: Underfitting score = {:.2f}\n"\
                #      .format(corrector, 
                #              self.metrics_list[cindex][corrector]['Under_fit']))
        
            print("PDC: Overfitting score  = {:.2f}"\
                  .format(self.metrics_list[cindex][corrector]['pdc_overfit']))
            print("SPOC PDC: Overfitting score  = {:.2f}"\
                  .format(
                self.metrics_list[cindex][corrector]['spoc_pdc_overfit']))
            #print("PDCSAP: Underfitting score = {:.2f}\n"\
            #      .format(self.metrics_list[cindex][corrector]['pdc_underfit']))

        if output=='overfit':
            self.hist_overfit(correctors,bins,alpha,density,spoc_pdc,
                              cumulative,histtype,log)
        if output=='sgcdpp':
            self.hist_sgcdpp(correctors,bins,alpha,density,spoc_pdc,
                             cumulative,histtype,log,
                             weights_combined_score)
        if output=='changes':
            print('Calculating and saving harmonic means of the sgCDPPs...')
            length=len(self.metrics_list)
            for i in np.arange(length):
                sgcdpp_list=[]
                sgcdpp_sap_list=[]
                sgcdpp_pdc_list=[]
                for corrector in correctors:
                    for minutes in minutes_list:
                        key = 'sgCDPP_{}min'.format(minutes)
                        key_sap = 'sap_sgCDPP_{}min'.format(minutes)
                        key_pdc = 'pdc_sgCDPP_{}min'.format(minutes)
                        sgcdpp = self.metrics_list[i][corrector][key]
                        sgcdpp_sap = self.metrics_list[i][corrector][key_sap]
                        sgcdpp_pdc = self.metrics_list[i][corrector][key_pdc]
                        sgcdpp_list.append(sgcdpp)
                        sgcdpp_sap_list.append(sgcdpp_sap)
                        sgcdpp_pdc_list.append(sgcdpp_pdc)
                    sgcdpp = statistics.harmonic_mean(sgcdpp_list)
                    sgcdpp_sap = statistics.harmonic_mean(sgcdpp_sap_list)
                    sgcdpp_pdc = statistics.harmonic_mean(sgcdpp_pdc_list)
                    self.metrics_list[i][corrector]['sgCDPP_harmonic_mean']=\
                    sgcdpp
                    self.metrics_list[i][corrector]['sap_sgCDPP_harmonic_mean']\
                    =sgcdpp_sap
                    self.metrics_list[i][corrector]['pdc_sgCDPP_harmonic_mean']\
                    =sgcdpp_pdc
                tic=self.cases_list[i]['TIC']
                sector=self.cases_list[i]['Sector']
                case_idx=self.cases_list[i]['Case_Index']
                self.save_metrics_to_json(save_name,overwrite_ok,
                                          self.metrics_list[i],tic,sector,
                                          case_idx)
                
        return
    
    def save_metrics_to_json(self,save_name,overwrite_ok,
                             metrics_dict,tic,sector,case_idx):
        """
        Saves metrics dictionaries to json files
        """
        
        path_to_pkl_file = '{}{}/{}.pkl'.format(
            dir_to_files,save_name,save_name)
        path_to_pkl_file, path_to_dir = self.set_file_dir(
            path_to_pkl_file,save_name,overwrite_ok)
        
        metrics_name = \
        'metrics_tic-{}-sector-{}-cindex-{}'.format(
            tic, sector, case_idx)
        path_to_json = path_to_dir+'json_files/metrics'
        makedirs(path_to_json,exist_ok=True)
        json_object = json.dumps(metrics_dict, indent = 4)
        with open("{}/{}.json".format(
            path_to_json,metrics_name), "w") as outfile:
            outfile.write(json_object)
            
        return
    
    def hist_overfit(self,correctors,bins,alpha,density,spoc_pdc,
                             cumulative,histtype,log):
        """
        Plots a histogram of the overfit score
        """
        
        length=len(self.metrics_list)
        
        self.overfit_list=[]
        labels=['PDC']
        colors=[self.parameters[labels[0]]['color']]
        
        overfit_pdc = np.zeros((length),dtype=float64)
        key = 'pdc_overfit'
        for i in np.arange(length):
            overfit_pdc[i] = self.metrics_list[i][correctors[0]][key]
        self.overfit_list.append(overfit_pdc)
        
        for corrector in reversed(correctors):
            overfit = np.zeros((length),dtype=float64)    
            key = 'Over_fit'
            for i in np.arange(length):
                overfit[i] = self.metrics_list[i][corrector][key]
            self.overfit_list.append(overfit)
            labels.append(corrector)
            colors.append(self.parameters[corrector]['color'])
        
        plt.figure()
        plt.hist(self.overfit_list,density=density,
                 bins=bins,alpha=alpha,label=labels,
                 cumulative=cumulative,histtype=histtype,log=log,color=colors)
        if spoc_pdc==True:
            overfit_spoc_pdc = np.zeros((length),dtype=float64) 
            key = 'spoc_pdc_overfit'
            for i in np.arange(length):
                overfit_spoc_pdc[i] = \
                self.metrics_list[i][correctors[0]][key]
            plt.hist(overfit_spoc_pdc,density=density,
                     bins=bins,alpha=alpha,label='SPOC PDC',color='orange',
                     cumulative=cumulative,histtype=histtype,log=log)
        plt.ylabel('Counts')
        plt.xlabel('Overfit Score')
        plt.legend(loc='upper left')
        
        return
    
    def hist_sgcdpp(self,correctors,bins,alpha,density,spoc_pdc,
                            cumulative,histtype,log,weights_combined_score):
        """
        Plots histograms for the sgCDPPs
        """
        
        length=len(self.metrics_list)
        
        for minutes in minutes_list:
            sgcdpp_list=[]
            labels=['PDC']
            colors=[self.parameters[labels[0]]['color']]
        
            sgcdpp_sap = np.zeros((length),dtype=float64)    
            key = 'sap_sgCDPP_{}min'.format(minutes)
            for i in np.arange(length):
                sgcdpp_sap[i] = self.metrics_list[i][correctors[0]][key]
            
            sgcdpp_pdc = np.zeros((length),dtype=float64)    
            key = 'pdc_sgCDPP_{}min'.format(minutes)
            for i in np.arange(length):
                sgcdpp_pdc[i] = self.metrics_list[i][correctors[0]][key]
                sgcdpp_pdc[i] = \
                ((sgcdpp_sap[i]-sgcdpp_pdc[i])/sgcdpp_sap[i])*100
            sgcdpp_list.append(sgcdpp_pdc)
            
            for corrector in reversed(correctors):
                sgcdpp = np.zeros((length),dtype=float64)    
                key = 'sgCDPP_{}min'.format(minutes)
                for i in np.arange(length):
                    sgcdpp[i] = self.metrics_list[i][corrector][key]
                    sgcdpp[i] = ((sgcdpp_sap[i]-sgcdpp[i])/sgcdpp_sap[i])*100
                sgcdpp_list.append(sgcdpp)
                labels.append(corrector)
                colors.append(self.parameters[corrector]['color'])
            
            plt.figure()
            plt.hist(sgcdpp_list,bins=bins,alpha=alpha,density=density,
                     cumulative=cumulative,histtype=histtype,log=log,
                     label=labels,color=colors,range=[-50,100])
            if spoc_pdc==True and minutes==60:
                sgcdpp_spoc_pdc = np.zeros((length),dtype=float64)
                key = 'pdc_CDPP_1h'
                for i in np.arange(length):
                    sgcdpp_spoc_pdc[i] = \
                    self.metrics_list[i][correctors[0]][key]*cds.ppm
                    sgcdpp_spoc_pdc[i] = \
                    ((sgcdpp_sap[i]-sgcdpp_spoc_pdc[i])/sgcdpp_sap[i])*100
                plt.hist(sgcdpp_spoc_pdc,bins=bins,alpha=alpha,density=density,
                         cumulative=cumulative,histtype=histtype,log=log,
                         label='SPOC PDC',color='orange',range=[-50,100])
            
            plt.ylabel('Counts')
            plt.xlabel('sgCDPP {} minutes (SAP-X)/SAP per cent'.format(minutes))
            plt.legend(loc='upper left')
        
#         plt.figure()
#         corr_i=0
#         for corr_label in labels:
#             plt.hist2d(sgcdpp_list[corr_i],self.overfit_list[0],bins=bins,
#                        norm=matcolors.LogNorm(),
#                        cmap=self.parameters[corr_label]['colormap'])
#             corr_i+=1
            
#             plt.colorbar(label='Counts for {}'.format(corr_label))
#             plt.ylabel('Overfit Score')
#             plt.xlabel('sgCDPP harmonic mean (SAP-X)/SAP per cent'.format(
#                 minutes))
#             #plt.legend(loc='upper left')
        
            fig, axs = plt.subplots(nrows=1, ncols=4, sharex=False, sharey=True, 
                                    constrained_layout=True)
            corr_i=0
            smin=0
            smax=100
            for ax, corr_label in zip(axs,reversed(labels)):
                ax.hist2d(sgcdpp_list[corr_i],
                          self.overfit_list[0],range=[[smin,smax],[0.5,1]],
                          bins=bins,norm=matcolors.LogNorm(),
                          cmap=self.parameters[corr_label]['colormap'])
                
                sfull=sgcdpp_list[corr_i]
                srange=sfull[(sfull)>smin & (sfull<smax)]
                ofull=self.overfit_list[0]
                orange=ofull[(sfull)>smin & (sfull<smax)]
                cc=np.corrcoef(srange,orange)
                #cc=np.corrcoef(sfull,ofull)
            
                ax.set_aspect(200,'box')
                ax.set_title('{} '.format(corr_label)+'\n'\
                             r'$\rho$'+' = {:.2f}'.format(cc[1,0]))

                if corr_i==0:
                    ax.set_xticklabels([0,50,100])
                    ax.set(ylabel='Overfit Score')
                else:
                    ax.set_xticklabels([None,50,100])
            
                corr_i+=1
        
            fig.subplots_adjust(wspace=0)
            fig.supxlabel('sgCDPP {} minutes (SAP-X)/SAP per cent'.format(
                minutes),y=0.25)     
        
        self.sgcdpp_mean_list=[]
        
        sgcdpp_mean_sap = np.zeros((length),dtype=float64)
        sgcdpp_mean_pdc = np.zeros((length),dtype=float64)
        key_sap = 'sap_sgCDPP_harmonic_mean'
        key_pdc = 'pdc_sgCDPP_harmonic_mean'
        for i in np.arange(length):
            sgcdpp_mean_sap[i] = self.metrics_list[i][correctors[0]][key_sap]
            sgcdpp_mean_pdc[i] = self.metrics_list[i][correctors[0]][key_pdc]
            sgcdpp_mean_pdc[i] = \
            ((sgcdpp_mean_sap[i]-sgcdpp_mean_pdc[i])/sgcdpp_mean_sap[i])*100
        self.sgcdpp_mean_list.append(sgcdpp_mean_pdc)
        
        for corrector in reversed(correctors):
            sgcdpp_mean = np.zeros((length),dtype=float64)
            sgcdpp_mean_sap = np.zeros((length),dtype=float64)
            sgcdpp_mean_pdc = np.zeros((length),dtype=float64)
            key = 'sgCDPP_harmonic_mean'
            key_sap = 'sap_sgCDPP_harmonic_mean'
            key_pdc = 'sap_sgCDPP_harmonic_mean'
            for i in np.arange(length):
                sgcdpp_mean[i]=self.metrics_list[i][corrector][key]
                sgcdpp_mean_sap[i]=self.metrics_list[i][correctors[0]][key_sap]
                sgcdpp_mean_pdc[i]=self.metrics_list[i][correctors[0]][key_pdc]
                sgcdpp_mean[i]=\
                ((sgcdpp_mean_sap[i]-sgcdpp_mean[i])/sgcdpp_mean_sap[i])*100
            self.sgcdpp_mean_list.append(sgcdpp_mean)
        
        plt.figure()
        plt.hist(self.sgcdpp_mean_list,bins=bins,alpha=alpha,density=density,
                 cumulative=cumulative,histtype=histtype,log=log,
                 label=labels,color=colors,range=[-50,100])
        plt.ylabel('Counts')
        plt.xlabel('sgCDPP (SAP-harmonic mean)/SAP per cent')
        plt.legend(loc='upper left')
        
        fig, axs = plt.subplots(nrows=1, ncols=4, sharex=False, sharey=True, 
                                constrained_layout=True)
        corr_i=0
        for ax, corr_label in zip(axs,reversed(labels)):
            ax.hist2d(self.sgcdpp_mean_list[corr_i],
                      self.overfit_list[0],range=[[smin,smax],[0.5,1]],
                      bins=bins,norm=matcolors.LogNorm(),
                      cmap=self.parameters[corr_label]['colormap'])
            
            sfull=sgcdpp_list[corr_i]
            srange=sfull[(sfull)>smin & (sfull<smax)]
            ofull=self.overfit_list[0]
            orange=ofull[(sfull)>smin & (sfull<smax)]
            cc=np.corrcoef(srange,orange)
            
            ax.set_aspect(200,'box')
            ax.set_title('{} '.format(corr_label)+'\n'\
                         r'$\rho$'+' = {:.2f}'.format(cc[1,0]))

            if corr_i==0:
                ax.set_xticklabels([0,50,100])
                ax.set(ylabel='Overfit Score')
            else:
                ax.set_xticklabels([None,50,100])
            
            #ax.set(xlabel='sgCDPP')
            
            #cbar = plt.colorbar(label='Counts for {}'.format(corr_label),
            #                    ticks=[1,2,3,4,5])
            #plt.clim(1, 5)
            #cbar.ax.set_yticklabels(["{:n}".format(i) for i in cbar.get_ticks()])
            
            corr_i+=1
        
        fig.subplots_adjust(wspace=0)
        fig.supxlabel('sgCDPP (SAP-harmonic mean)/SAP per cent',y=0.25)
        
        self.hist_combined_score(correctors,bins,alpha,density,spoc_pdc,
                                 cumulative,histtype,log,labels,colors,
                                 weights_combined_score)
        
        return

    def hist_combined_score(self,correctors,bins,alpha,density,spoc_pdc,
                            cumulative,histtype,log,labels,colors,
                            weights_combined_score):
        """
        Plots histograms for an overall combined metric
        
        Optional parameter: 
                    weights_combined_score: array-like, with the weights 
                                            for the harmonic mean combining
                                            the mean sgdcpp (first element)
                                            and overfit metrics (second element)
        """
        
        length=len(self.metrics_list)
        
        combined_score_list=[]
        sgcdpp_mean_sigmoid_list = []
        
        for sgcdpp_mean in self.sgcdpp_mean_list:
            sgcdpp_mean_sigmoid_list.append(self.sigmoid(sgcdpp_mean/100))
        
        list_i=0
        for overfit in self.overfit_list:
            combined_score = np.zeros((length),dtype=float64)
            for arr_i in np.arange(length):
                sig = sgcdpp_mean_sigmoid_list[list_i][arr_i]
                ofit = self.overfit_list[list_i][arr_i]
                data = np.array([sig,ofit])
                combined_score[arr_i] = statistics.harmonic_mean(
                    data,weights=weights_combined_score)
            combined_score_list.append(combined_score)
            list_i+=1
        
        plt.figure()
        plt.hist(combined_score_list,bins=bins,alpha=alpha,density=density,
                 cumulative=cumulative,histtype=histtype,log=log,
                 label=labels,color=colors)
        plt.ylabel('Counts')
        plt.xlabel('Combined Score'.format(minutes))
        plt.legend(loc='upper left')
        
        return
    
    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))
    
    def twiceSigmoidInv(self,x):
        return 2 / (1 + np.exp(x))