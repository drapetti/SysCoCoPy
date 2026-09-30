"""
File: InSimPy/Source/Simulations_Injector.py
Author: David Rapetti
Date: 12 Aug 2023

Description: Injects stellar variability simulations at the pixel level
             into a target pixel file (TPF) taking into account the TESS PRF 
             (Pixel Response Function).
"""

import PRF
import lightkurve as lk
import numpy as np
import sys, os, copy, gc
import dill as pickle
import matplotlib.pyplot as plt
from os import makedirs
from astropy import units as u
from astropy.units import cds, Quantity
from astropy.io import fits
from IPython.display import clear_output

import SysCoCoPy.Source.Correctors_Comparer as cc
from SysCoCoPy.Source.Util.Target_Lists.Targets_file_reader \
    import read_target_cases, download_target_files

"""
If specific quality and transit masks are needed, the following 
quantities would needed to be provided for each target.
"""

#quality_bitmask = 16575   # original from JT
quality_bitmask = 16575+512 # from JT+filter out SPOC outliers from lc/tpf
#quality_bitmask = 17071    # test requested by JT for TOI-1835
#quality_bitmask = 16431   # additional test requested by JT, not used
##quality_bitmask = ( 4 | 16 | 32 ) # from H21

current_directory=os.getcwd()
insimpy_source_dir='/InSimPy/Source/'
syscocopy_source_dir='/SysCoCoPy/Source/'
dir_to_files=current_directory+syscocopy_source_dir+'data_files/saved_output/'

#class Simulations_Injector(CC):
class Simulations_Injector():
    """
    0) Read 2-min TPFs
    1) Obtain TESS PRFs in the postage stamps of the targets
    2) Generate Stellar Variability Model (SVM)
    3) Include SVM at the pixel level
    4) Save the new TPFs as fits files
    """
    
    def __init__(self, targets, model, model_params):
        """
        This constructor collects input for simulation injections:
            a) target/sector file names, 
            b) SV model
        """
        
        self.targets = targets
        self.model = model
        self.model_params = model_params
        
        return
    
    def read_cases(self,save_name=None,overwrite_ok=False,download=True):
        """
        Reads in the required tpf filenames for the case list.
        
        Optional parameters:
        
        save_name    : sets the name of the directory storing the fits files
                       of the new tpfs. The same name is used for the pkl file 
                       storing the cases object
        overwrite_ok : set True if you are ok with overwriting files and
                       directories and want to avoid questions about it
        download     : set False if there is no need for downloading data files
        """
        
        if save_name == None:
            save_name=input('Enter name of saving directory: ')
        
        path_to_pkl_file = '{}{}/{}.pkl'.format(
            dir_to_files,save_name,save_name)
        path_to_pkl_file, path_to_dir = cc.set_file_dir(
            cc,path_to_pkl_file,save_name,overwrite_ok)
        
        if download == True:
            cc.cases_list = read_target_cases(self.targets)
            cc.spoc_lc_filenames, cc.tpf_filenames, cc.quat_list, \
            cc.cbv_dir = download_target_files(
                cc.cases_list, quality_bitmask)
            with open(path_to_pkl_file, 'wb') as pickle_file:
                pickle.dump(cc.spoc_lc_filenames, pickle_file) 
                pickle.dump(cc.tpf_filenames, pickle_file)
                pickle.dump(cc.quat_list, pickle_file)
                pickle.dump(cc.cbv_dir, pickle_file)
                pickle.dump(cc.cases_list, pickle_file)
        else:
            cc.load_saved_input(cc,save_name=save_name)
        
        return
    
    def inject(self,save_name=None,overwrite_ok=False,download=True,
               plot=False,null_test=None):
        """
        - Reads in the TPFs and finds the PRFs at a given cadence
        - Using the PRF at the poining, injects a stellar variability model 
        into the TPF to provide a simulated TPF
        
        Optional parameters:
        
        save_name, overwrite_ok, download : work as in cc.compare
        
        plot      : plots the prf as a function of cadence
        null_test : plots the resampled prf minus the tpf at the given 
                    array position 
        
        """
        
        self.read_cases(save_name,overwrite_ok,download)
        
        self.tpf_inj_filenames = []
        
        cindex = 0
        for case in cc.cases_list:
            tic = case['TIC']
            sector = case['Sector']
            case_idx = case['Case_Index']
            
            #Reads TPFs
            tpf = lk.read('{}'.format(cc.tpf_filenames[cindex]),
                        quality_bitmask=quality_bitmask)
            
            #Reads input in TPFs
            
            #column axis
            val_ref_ccd_col_1 = tpf.hdu[2].header['CRVAL1P'] #lower-left in ccd
            ref_pix_axis_1 = tpf.hdu[2].header['CRPIX1'] #position in tpf
            ref_ccd_col_1 = tpf.hdu[2].header['CRPIX1P'] #FITS origin
            col_pos_corr_1 = tpf.hdu[1].data['POS_CORR1'] #pointing corrections
            
            #the same for row axis
            val_ref_ccd_col_2 = tpf.hdu[2].header['CRVAL2P']
            ref_pix_axis_2 = tpf.hdu[2].header['CRPIX2']
            ref_ccd_col_2 = tpf.hdu[2].header['CRPIX2P']
            col_pos_corr_2 = tpf.hdu[1].data['POS_CORR2']
            
            #tpf pixels in axes
            naxis_1 = tpf.hdu[2].header['NAXIS1']
            naxis_2 = tpf.hdu[2].header['NAXIS2']
            
            cam = tpf.hdu[0].header['CAMERA']
            ccd = tpf.hdu[0].header['CCD']
            sector = tpf.hdu[0].header['SECTOR']
            
            #tpf flux
            flux = tpf.hdu[1].data['FLUX']
            
            #Calculates reference pixel column and row in CCD
            colnum = val_ref_ccd_col_1 + ref_pix_axis_1 - ref_ccd_col_1 
            rownum = val_ref_ccd_col_2 + ref_pix_axis_2 - ref_ccd_col_2 
                        
            #print('colnum = ',colnum,'rownum = ',rownum)
            
            prf = PRF.TESS_PRF(cam,ccd,sector,colnum,rownum)
            
            tpf_inj = copy.deepcopy(tpf)
            tpf = copy.deepcopy(tpf)
            
            for indx in range(4):
                tpf_inj.hdu[indx].header['SIMDATA'] = True
            
            if null_test != None:
                    #lc = tpf.to_lightcurve()
                    #flux_test = lc.flux(null_test)   
                    flux_test = flux[null_test].sum()
                    
            tcount=0
            for pos in col_pos_corr_1:
                
                stamp_pos_1 = ref_pix_axis_1
                stamp_pos_2 = ref_pix_axis_2
                if np.isnan(pos) != True:
                    #stamp pointing corrections per cadence
                    stamp_pos_1 += col_pos_corr_1[tcount] - 1.0
                    stamp_pos_2 += col_pos_corr_2[tcount] - 1.0                    
                
                #Resamples the PRF to the center of the TPF
                prf_resampled = prf.locate(stamp_pos_1,stamp_pos_2,
                                           (naxis_1,naxis_2))                
                    
                if tcount == null_test:
                    tpf_test = tpf.hdu[1].data['FLUX'][tcount]
                    prf_test = flux_test*prf_resampled.sum()*prf_resampled
                    percent_diff = ((tpf_test-prf_test)/flux_test)*100
                    self.plot_prf(percent_diff)
                
                sim_injection = prf_resampled * self.SV_model(tcount)
                
                tpf_inj.hdu[1].data['FLUX'][tcount] += sim_injection

                if plot==True:
                    self.plot_prf(sim_injection)
                    
                tcount+=1
            
            mvalstr=str(self.model_params[self.model].values())
            mval=mvalstr.replace(
                'dict_values([','').replace('])','').replace(', ','-')
            tpf_inj_filename = cc.tpf_filenames[cindex].replace(
                'tp.fits','sMod-{}_sPar-{}_tp.fits'.format(self.model,mval))
            
            self.tpf_inj_filenames.append(tpf_inj_filename)
            
            #Saves to a FITS file
            tpf_inj.to_fits(tpf_inj_filename,overwrite=True)
            
            cindex += 1
            
            gc.collect()
        
        return
        
    def SV_model(self,tcount):
        """
        Generates the chosen stellar variability model
        """
            
        if self.model == 'sinusoidal':
            
            amplitude = self.model_params[self.model]['amplitude'] #e-/s
            period = self.model_params[self.model]['period'] #in 2min cadences
            phase = self.model_params[self.model]['phase'] #in degrees
            
            sv_model = amplitude * np.sin(((tcount)*2*np.pi/period)
                                          -(np.pi/180)*phase)
            
        if self.model == 'double-sinusoidal':
            
            amplitude1 = self.model_params[self.model]['amplitude1'] #e-/s
            period1 = self.model_params[self.model]['period1'] #in 2min cadences
            phase1 = self.model_params[self.model]['phase1'] #in degrees
            
            amplitude2 = self.model_params[self.model]['amplitude2'] #e-/s
            period2 = self.model_params[self.model]['period2'] #in 2min cadences
            phase2 = self.model_params[self.model]['phase2'] #in degrees
            
            sinusoidal1 = amplitude1 * np.sin(((tcount)*2*np.pi/period1)
                                          -(np.pi/180)*phase1)
            sinusoidal2 = amplitude2 * np.sin(((tcount)*2*np.pi/period2)
                                          -(np.pi/180)*phase2)
            
            sv_model = sinusoidal1 + sinusoidal2
            
        return sv_model

    def plot_prf(self,sim_injection):
        """
        Plots the postage stamp of the prf for the reference pixel
        """
        
        #Plots PRF in the center of the TPF
        clear_output(wait=True)
        plt.imshow(sim_injection)
        ax = plt.gca()
        ax.invert_yaxis()
        plt.colorbar()
        plt.show()
        
        return
    