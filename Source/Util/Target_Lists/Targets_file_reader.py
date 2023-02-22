"""
File: SysCoCoPy/Util/Target_Lists/Targets_file_reader.py
Author: David Rapetti
Date (second version): 06 Apr 2022
Date (third version) : 22 Jun 2022

Description: Methods to download and read target cases for the
             Corrector Comparer class
"""

from astropy.utils.decorators import deprecated
import lightkurve as lk

author='SPOC'
exptime=120

def read_target_cases(targets_file):
    """
    Reads the input csv file into the corresponding returned list.
    
    Input parameter:
    
    targets_file    : csv file with the selection list of targets and, 
                      optionally, sectors (if none is specified, all sectors
                      with that target are included)
    """
    
    import numpy as np
    from csv import reader
    from os.path import exists
    from os import system
    
    download_dir = 'Util/Target_Lists/data_files/Lists/'
    
    global_cindex = 0

    cases_list = []
    
    #reads csv file of cases
    with open('{}{}'.format(download_dir,targets_file)) as csvfile:
        csvtargets = reader(csvfile, delimiter=',')
        header = next(csvtargets)
        for target in csvtargets:
            tic = 'TIC {}'.format(target[0])
            if len(target)==1:
                search = lk.search_lightcurve(
                    tic, author=author, exptime=exptime)
                sectors = [int(search[x].mission[0].split()[-1]) \
                         for x in np.arange(len(search))]
            else:
                sectors = target[1:]
            for sector in sectors:
                #builds the cases list
                cases_list.append(
                    {'TIC':'{}'.format(tic.split(' ')[-1]),
                     'Sector':'{}'.format(sector),
                     'Case_Index':'{}'.format(global_cindex+1)})                
                global_cindex += 1
                   
    return cases_list

def download_target_files(cases_list,quality_bitmask):
    """
    Searches and downloads light curves (lc), target pixel files (tpf),
    quaternions and CBV files from the MAST portal.
    
    Parameters:
    
    cases_list      : list of cases as read by the method read_target_cases
    quality_bitmask : quality bitmask to obtain lcs from tpfs
    author          : To choose the mission in searching the lc. 
                      By default is TESS ('SPOC')
    exptime         : To choose the exposure time in searching the lc.
                      By default is 2 minutes.
    """
    
    import requests
    import re
    import urllib.request
    import numpy as np
    import pathlib
    from os.path import exists
    from os import system
    
    download_dir = 'Util/Target_Lists/data_files/Lists/'
    quatdir = download_dir + "stsciDownload/TESS/Quaternions/"
    cbvdir = download_dir + "cbvDownload/TESS/"
    cbvfits_dir = cbvdir + "fits_files/"
    
    urlBase = "https://archive.stsci.edu/missions/tess/"
    quat_urldir = urlBase + "engineering/"
    cbv_urldir = urlBase + "download_scripts/sector/"    

    quat_request = requests.get(quat_urldir)
    quat_files_all = re.findall(r'tess.*?quat.fits',quat_request.text)
    
    cbv_request = requests.get(cbv_urldir)
    cbvsh_files = re.findall(r'tesscurl_sector.*?_cbv.sh',cbv_request.text)

    spoc_lc_list = []
    spoc_lc_filenames = []
    tpf_list = []
    tpf_filenames = []
    quat_list = []

    for case in cases_list:
        tic = 'TIC {}'.format(case['TIC'])
        sector = case['Sector']
        
        #searches lcs and tpfs
        search_lc = lk.search_lightcurve(
            tic, author=author, sector=sector, exptime=exptime)
        search_tpf = lk.search_targetpixelfile(
            tic, author=author, sector=sector, exptime=exptime) 
        
        #downloads lcs and tpfs if not already done so
        spoc_lc_filename = search_lc.download(
            download_dir=download_dir,
            quality_bitmask=quality_bitmask).meta['FILENAME']
        search_tpf.download(
            download_dir=download_dir,quality_bitmask=quality_bitmask)
        tpf_filename = spoc_lc_filename.replace('lc.fits','tp.fits')
        spoc_lc_filenames.append(spoc_lc_filename)
        tpf_filenames.append(tpf_filename)
        
        #downloads quaternions if not already done so
        for file in quat_files_all:
            sec_num=int(
            file.split('-')[0].split('_')[1].split('sector')[1])
            if sec_num==int(sector):
                url = quat_urldir + file
                quat_file = quatdir + file
                quat_file_exists = exists(quat_file)
                if quat_file_exists == False:
                    urllib.request.urlretrieve(url, quat_file)
        quat_list.append(quat_file)
                
        #downloads CBVs if not already done so
        for file in cbvsh_files:
            sec_num=int(file.split('_')[2])
            if sec_num==int(sector):
                url = cbv_urldir + file
                cbv_file = cbvdir + file
                cbv_file_exists = exists(cbv_file)
                if cbv_file_exists == False:
                    #TODO: TEST EXCEPTION
                    try:
                        urllib.request.urlretrieve(url, cbv_file)
                        system('chmod +x {}'.format(cbv_file))
                        system('(cd {} && "{}")'.format(
                            cbvfits_dir, cbv_file))
                    except:
                        system('rm {}'.format(cbv_file))
                        print("Download processing interrupted.")
                        print("Last shell-file removed: {}".format(
                            cbv_file))
       
    #return spoc_lc_list, tpf_list, quat_list, cbvfits_dir
    return spoc_lc_filenames, tpf_filenames, quat_list, cbvfits_dir

@deprecated(since="1.0")
def read_target_file(json_name,quality_bitmask,output='filenames',
                     filetype='tp',printing=False,output_flag=False):
    '''
    Reads a json file of a target cases list within the directory "data_files/"
    
    Note: the target cases json files (using the format below, i.e. a dictionary 
    with labels 'TOI', 'TIC', 'Sector', 'filenames', with the latter
    being in turn a dictionary with labels 'tp', 'lc, 'quat') are built using
    the jupyter notebook 'Targets_file_generator.ipynb'.
    
    Returns the corresponding python lists 
        a) SPOC light curves,
        b) tpf files,
        c) quaternion file names,
        d) targets
    
        json_name       : name of the json file without the extension
        quality_bitmask : selects the quality flags apply to the data
        
        output to print  : 'TOI'       TESS Object of Interest number
                           'TIC'       TESS Input Catalog number
                           'Sector'    TESS sector number
                           'filenames' See filetype below
        
        filetype to print : 'tp'        target pixel files 
                            'lc'        light curves
                            'quat'      spacecraft quaternions
                    
        printing    : Boolean to print out results
        
        output_flag : Boolean to select returning only SPOC light curves
                      and target list
    '''
    import json
    import os
    import numpy as np
    import lightkurve as lk
    
    cwd = os.getcwd()
    
    target_path = cwd + '/Util/Target_Lists/data_files/'

    tpf_path  = cwd + '/data_files/tpfs/'
    lc_path   = cwd + '/data_files/lcs/'
    quat_path = cwd + '/data_files/quaternions/'
    
    with open(target_path + '{}.json'.format(json_name)) as Target_list_file:
        Target_list = json.load(Target_list_file)
    
    targets = np.arange(len(Target_list))
    
    spoc_lc_list = []
    tpf_list = []
    quat_list = []

    for target in targets:
        if printing == True:
            if output != 'filenames':
                print(Target_list[target]['{}'.format(output)])
            else:
                print(Target_list[target]['filenames']['{}'.format(filetype)])
                
        tpf_file  = tpf_path  + Target_list[target]['filenames']['tp']
        lc_file   = lc_path   + Target_list[target]['filenames']['lc']
        quat_name = quat_path + Target_list[target]['filenames']['quat']
        
        spoc_lc = lk.read(lc_file, quality_bitmask=quality_bitmask)
        spoc_lc_list.insert(target, spoc_lc)
        
        tpf = lk.read(tpf_file, quality_bitmask=quality_bitmask,
                      aperture='pipeline')
        tpf_list.insert(target, tpf)
        
        quat_list.insert(target, quat_name)
    
    if output_flag==False:
        return spoc_lc_list, tpf_list, quat_list, Target_list
    else:
        return spoc_lc_list, Target_list
