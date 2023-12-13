"""
File: SysCoCoPy/Util/Target_Lists/Targets_file_reader.py
Author: David Rapetti
Date (second version): 06 Apr 2022
Date (third version) : 22 Jun 2022

Description: Methods to download and read target cases for the
             Corrector Comparer class
"""

import lightkurve as lk
from astropy.utils.decorators import deprecated
from os import getcwd

author_default='SPOC'
exptime_default=120

current_directory = getcwd()
download_dir = current_directory + \
'/SysCoCoPy/Source/Util/Target_Lists/data_files/Lists/'
    
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
    
    global_cindex = 0

    cases_list = []
    
    #reads csv file of cases
    with open('{}{}'.format(download_dir,targets_file)) as csvfile:
        csvtargets = reader(csvfile, delimiter=',')
        header = next(csvtargets)
        if header[0] == 'Author':
            author_time = next(csvtargets)
            author = author_time[0]
            exptime = author_time[1]
            header2 = next(csvtargets)
        else:
            author = author_default
            exptime = exptime_default
        
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
                     'Author':author,
                     'Exptime':exptime,
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
                      By default is 2 minutes
    """
    
    import requests
    import re
    import urllib.request
    import numpy as np
    import pathlib
    import tarfile
    from os.path import exists
    from os import system
    
    #Download directories for:
    #Quaternions
    quatdir = download_dir + "stsciDownload/TESS/Quaternions/"
    #CBV bundles
    cbvdir = download_dir + "cbvDownload/TESS/"
    cbvdir_hlsp = donwload_dir + "cbvDownload/HLSP/"
    #CBV FITS files
    cbvfits_dir = cbvdir + "fits_files/"
    cbvfits_dir_hlsp = cbvdir_hlsp + "fits_files/"
    
    #MAST directory urls:
    urlBase = "https://archive.stsci.edu/"
    urlBase_TESS = urlBase + "missions/tess/"
    #Quaternions
    quat_urldir = urlBase_TESS + "engineering/"
    #SPOC CBVs
    cbv_urldir = urlBase_TESS + "download_scripts/sector/"
    #TESS-SPOC CBVs
    cbv_urldir_hlsp = urlBase + "hlsp/tess-spoc/"
    
    #MAST files:
    #SPOC CBV curls
    curlBaseFile = "tesscurl_sector_"
    curlEndFile = "_cbv.sh"
    #TESS-SPOC CBV tar.gzs
    gzEndFile = "_tess_v01_cbv-bulk-dl.tar.gz"
    
    quat_request = requests.get(quat_urldir)
    quat_files_all = re.findall(r'tess.*?quat.fits',quat_request.text)
    
    spoc_lc_filenames = []
    tpf_filenames = []
    quat_list = []

    for case in cases_list:
        tic = 'TIC {}'.format(case['TIC'])
        sector = case['Sector']
        author = case['Author']
        exptime = case['Exptime']

        #For SPOC CBVs
        cbv_curl_file = curlBaseFile + str(sector) + curlEndFile
        #For TESS-SPOC CBVs
        gzSectorUrl = "s%04d/hlsp_tess-spoc_tess_ffi_s%04d" % (sector, sector)
        cbv_gz_file = cbv_urldir_hlsp + gzSectorUrl + gzEndFile
        
        #Searches and downloads lcs and tpfs if not already done so
        spoc_lc_filename = lk.search_lightcurve(
            tic, author=author, sector=sector, exptime=exptime).download(
            download_dir=download_dir,
            quality_bitmask=quality_bitmask).meta['FILENAME']
        tpf_filename = lk.search_targetpixelfile(
            tic, author=author, sector=sector, exptime=exptime).download(
            download_dir=download_dir,quality_bitmask=quality_bitmask).path
        
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
                
        #downloads SPOC CBVs if not already done so
        if author == 'SPOC':
            cbv_url_file = cbv_urldir + cbv_curl_file
            cbv_loc_file = cbvdir + cbv_curl_file
            cbv_file_exists = exists(cbv_url_file)
            if cbv_file_exists == False:
                #TODO: TEST EXCEPTION
                try:
                    urllib.request.urlretrieve(cbv_url_file,cbv_loc_file)
                    system('chmod +x {}'.format(cbv_loc_file))
                    system('(cd {} && "{}")'.format(
                        cbvfits_dir, cbv_loc_file))
                except:
                    system('rm {}'.format(cbv_loc_file))
                    print("Downloading process interrupted.")
                    print("Last .sh file removed: {}".format(cbv_loc_file))
        elif author == 'TESS-SPOC':
            cbv_url_file = cbv_urldir_hlsp + cbv_gz_file
            cbv_loc_file = cbvdir_hlsp + cbv_gz_file
            cbv_file_exists = exists(cbv_url_file)
            if cbv_file_exists == False:
                try:
                    urllib.request.urlretrieve(cbv_url_file,cbv_loc_file)
                    file = tarfile.open(cbv_loc_file)
                    file.extractall(cbvfits_dir_hlsp)
                    file.close()
                except:
                    system('rm {}'.format(cbv_loc_file))
                    print("Downloading process interrupted.")
                    print("Last tar.gz file removed: {}".format(cbv_loc_file))
        else:
            raise Exception('The author selected, ({}), is currently not'\
                            'accepted by the CBV corrector.'.format(author))

    return (spoc_lc_filenames, tpf_filenames, quat_list, 
            cbvfits_dir, cbvfits_dir_hlsp)

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
