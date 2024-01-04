import numpy as np
import copy
from matplotlib import pyplot as plt

from data_preprocess.tool_code.matlab_python.addhw import addhw
from data_preprocess.tool_code.matlab_python.ds2frns import ds2frns


def ds2procds(ds):
    ds = copy.deepcopy(ds)
    strarr = []   # Error/Warning string 
    # Check SIMULATED Data:
    if 'DataType' in ds:
        if ds['DataType'] == 1:
            pds = {}
            pds['Data'] = 1e6 * ds['MainData']
            Sy, Sx, N = pds['Data'].shape
            pds['Time'] = ds['MainTime']
            pds['Period'] = ds['Period']
            pds['SatInds'] = [None] * ds['Period']
            pds['LEDLocsXY'] = ds['LEDLocsXY']
            pds['CmPerPixel'] = ds['CmPerPixel']
            pds['Pressure'] = ds['MainPressure']
            pds['PressureTime'] = ds['MainPressureTime']
            pds['DarkDiffMeanMax'] = 1e5
            pds['DarkDiffStdMax'] = 1e5
            pds['DarkDiff'] = np.zeros(ds['DarkData'].shape)
            pds['ProcessingModeds'] = ds['ProcessingModeds']
            pds['DataType'] = ds['DataType']
            pds['MarkerP1'] = ds['MarkerP1']
            pds['MarkerP2'] = ds['MarkerP2']
            return pds, strarr
    
    # Cut SQR Frames:
    VeriFlag = 0
    if 'Verification' in ds:
        VeriFlag = ds['Verification']
    
    if VeriFlag != 1:
        d_und=ds['Main_image']
        ii = np.where(ds['Main_stat'] == 8)[0]
        ds['Main_image'] = np.delete(ds['Main_image'], ii, axis=0)
        ds['MainTime'] = np.delete(ds['MainTime'], ii)
        ds['Main_stat'] = np.delete(ds['Main_stat'], ii)
        ds['MainPressure'] = np.delete(ds['MainPressure'], ii)
        ds['MainPressionTime'] = np.delete(ds['MainPressionTime'], ii)
        ds['ILState'] = np.delete(ds['ILState'], ii, axis=1)
        d, d1, d2, d3, d4, d5 = ds['Main_image'],ds['MainTime'],ds['Main_stat'],ds['MainPressure'],ds['MainPressionTime'],ds['ILState']#到这输入的帧数是一致的
    ds = addhw(ds)  # Add Hardware Fields这部分不需要写！
    TotalFrames, Sy, Sx = ds['Main_image'].shape  # Input Data Size
    
    # Check if the Scan is complete:
    # for k in range(TotalFrames - 1, -1, -1):
    #     img = ds['Main_image'][:, :, k]
    #     if np.sum(np.abs(img)) == 0:
    #         raise ValueError('Scan Aborted')
    
    # Subsequence to load:
    fds, strarr = ds2frns(ds)
    Select = fds['Select']
    Period = fds['Period']
    #
    # Experimental Subframes (before Jump, after 2-nd Jump, etc.):
    Select2 = []
    if 'Select2' in fds:
        Select2 = fds['Select2']
    #

    # Illuminator Stuff:
    dl=ds['LevLED']
    ds['LevLED'] = ds['LevLED'][Select-1]
    Linds = np.where(ds['LevLED'] <= 0)[0]
    if len(Linds) > 0:
        return ds,strarr
        # raise ValueError('LED Error: Non-positive LED Level!')
    #
    # # Convert Data to Subsequence:
    TotalFrames = len(Select)
    if Select2:
        ds['Data2'] = ds['MainData'][:, :, Select2]
    ds['Main_image'] = ds['Main_image'][Select-1,:, :]
    ds['MainTime'] = ds['MainTime'][Select-1]
    ds['MainPressure'] = ds['MainPressure'][Select-1]
    ds['MainPressionTime'] = ds['MainPressionTime'][Select-1]

    # 检查是否饱和或前后暗帧 1为不合格
    # Saturation Check:
    if Sy == 102 and Sx ==128:
        SatThld = 4090
    elif Sy == 161 and Sx == 202:
        SatThld = 65000

    ds['SaturationCheck'] = 0
    for k in range(Period):
        Inds = np.where(np.sum(ds['Main_image'][k::Period, :, :] > SatThld, axis=0) > 0)
        if len(Inds[0]) > 30:
            ds['SaturationCheck'] = 1
            break

    if ds['HWVersion'] == 2:
        DarkDiffMeanMax = 1.5
        DarkDiffStdMax = 2.5
        if ds['ProcessingModeds']['NIRImage'] == 1:
            DarkDiffMeanMax = 3.0
    elif ds['HWVersion'] == 3:
        DarkDiffMeanMax = 10
        DarkDiffStdMax = 50
    else:
        DarkDiffMeanMax = 10
        DarkDiffStdMax = 10

    # 与matlab 有些不同，Matlab只检查有效光照区域
    if np.abs(np.mean(ds['DarkDiff'])) > DarkDiffMeanMax or np.std(ds['DarkDiff'] > DarkDiffStdMax):
        ds['DarkCheck'] = 1
    else:
        ds['DarkCheck'] = 0

    ds['Main_image'] = ds['Main_image'].astype(np.float32)
    # Subtract Dark Frame:
    for k in range(TotalFrames):
        ds['Main_image'][k, :, :] = ds['Main_image'][k, :, :] / float(ds['LevLED'][k])
        dres = ds['Main_image'][k, :, :]

    # Divide by LED Intensity:
    for k in range(TotalFrames):
        cha = ds['Main_image'][k, :, :].astype(np.float32) - ds['Dark_image'].astype(np.float32)
        ds['Main_image'][k, :, :] = cha

    
    return ds, strarr