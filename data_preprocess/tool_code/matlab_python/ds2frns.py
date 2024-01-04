import numpy as np
from sympy import *


def frns2select(FirstNumbers, Step, Count, TotalFrames=None):
    if TotalFrames is not None:
        Nmax = np.max(FirstNumbers)
        Countmax = 1 + np.floor((TotalFrames - Nmax) / Step)
        if Count > Countmax:
            print('Number of cycles loaded:', Countmax)
        Count = min(Count, Countmax)

    Period = len(FirstNumbers)
    Select = np.zeros(Period * Count, dtype=int)
    for s in range(Count):
        Select[s * Period:(s + 1) * Period] = FirstNumbers + s * Step

    return Select


def ds2frns(ds, Mode=0):
    strarr = []
    Mode = 0
    fds = {}
    TotalFrames, Sy, Sx = ds['Main_image'].shape
    LEDPeriod = ds['LEDPeriod']

    # DEFAULT (Unrecognized Protocol) Frames:
    N = TotalFrames // LEDPeriod  # 到这为止都是对的上的
    ii = np.arange(LEDPeriod)  # 索引不同
    if N > 1:
        dsout_Select = np.concatenate([ii, ii + (N - 1) * LEDPeriod])  # 选择帧数的索引和matlab差1，可行？
    else:
        dsout_Select = ii
    dsout_Period = LEDPeriod

    # Define Subsequence Numbers:
    if ds['HWVersion'] == 1:  # "Slow" Apogee Camera
        if TotalFrames == 120:  # 3 LED Protocol Identified
            dsout_Select = frns2select([34, 35, 36], LEDPeriod, 19)
            dsout_Period = 3

    elif ds['HWVersion'] == 2 or ds['HWVersion'] == 3:  # Full-Field Roper Camera &QI815
        # if ds['ProcessingModeds']['NIRImage'] == '1':
        if '1' in ds['ProcessingModeds']['NIRImage']:
            if LEDPeriod == 3 and TotalFrames == 84:  # NIR breast dual wave 3LED
                dsout_Select = frns2select(np.array([10, 11, 12]) + 6 + 75 * Mode, LEDPeriod, 23, TotalFrames)
                dsout_Period = 3
            elif LEDPeriod == 5 and TotalFrames == 90:  # NIR breast dual wave 5LED
                dsout_Select = frns2select(np.array([11, 12, 13, 14, 15]) + 15 + 75 * Mode, LEDPeriod, 13, TotalFrames)
                dsout_Period = 5
        elif 'twowaveDcm' in ds['ProcessingModeds']:
            if LEDPeriod == 3 and TotalFrames == 156:  # 3 LED protocol for smaller breasts
                dsout_Select = frns2select(np.array([31, 32, 33]) + 6 + 75 * Mode, LEDPeriod, 23, TotalFrames)
                dsout_Period = 3
            elif LEDPeriod == 5 and TotalFrames == 160:  # 5 LED protocol for bigger breasts
                dsout_Select = frns2select(np.array([26, 27, 28, 29, 30]) + 15 + 75 * Mode, LEDPeriod, 13, TotalFrames)
                dsout_Period = 5
        elif 'PressureTimeC' in ds['ProcessingModeds']:  # 单波段700ms扫描间隔
            if ds['ProcessingModeds']['PressureTimeC'] == 2:
                if LEDPeriod == 3 and TotalFrames == 156:  # 3 LED protocol for smaller breasts
                    # Pressure Jumps after (36, and 111) + 6
                    dsout_Select = frns2select(np.array([43, 44, 45]) + 6 + 75 * Mode - 3 * 4, LEDPeriod, 23,
                                               TotalFrames)
                    dsout_Period = 3
                elif LEDPeriod == 5 and TotalFrames == 160:  # 5 LED protocol for bigger breasts
                    # Pressure Jumps after (35, and 110) + 10
                    dsout_Select = frns2select(np.array([41, 42, 43, 44, 45]) + 15 + 75 * Mode - 3 * 5, LEDPeriod, 13,
                                               TotalFrames)
                    dsout_Period = 5
        else:
            if LEDPeriod == 3 and TotalFrames == 150:  # 3 LED protocol for smaller breasts
                # Pressure Jumps after (36, and 111)
                dsout_Select = frns2select(np.array([43, 44, 45]) + 75 * Mode, LEDPeriod, 23, TotalFrames)
                dsout_Period = 3
            elif LEDPeriod == 5 and TotalFrames == 150:  # 5 LED protocol for bigger breasts
                # Pressure Jumps after (35, and 110)
                dsout_Select = frns2select(np.array([41, 42, 43, 44, 45]) + 75 * Mode, LEDPeriod, 14, TotalFrames)
                dsout_Period = 5
            elif LEDPeriod == 3 and TotalFrames == 156:  # 3 LED protocol for smaller breasts
                # Pressure Jumps after (36, and 111) + 6
                res = np.array([43, 44, 45]) + 6 + 75 * Mode
                dsout_Select = frns2select(np.array([43, 44, 45]) + 6 + 75 * Mode, LEDPeriod, 23, TotalFrames)
                # dsout_Select = frns2select([43, 44, 45] + 6 + 75 * Mode, LEDPeriod, 23, TotalFrames)
                dsout_Period = 3
            elif LEDPeriod == 5 and TotalFrames == 160:  # 5 LED protocol for bigger breasts
                # Pressure Jumps after (35, and 110) + 10
                dsout_Select = frns2select(np.array([41, 42, 43, 44, 45]) + 15 + 75 * Mode, LEDPeriod, 13, TotalFrames)
                dsout_Period = 5
            elif LEDPeriod == 3 and TotalFrames == 303:  # 3 LED protocol for smaller breasts
                # Pressure Jumps after 42, 117, and 192
                dsout_Select = frns2select(np.array([43, 44, 45]) + 6 + 75 * Mode, LEDPeriod, 23, TotalFrames)
                dsout_Period = 3
            elif LEDPeriod == 5 and TotalFrames == 305:  # Dual Jump protocol for bigger breasts
                # Pressure Jumps after 45, 120,  and 195
                dsout_Select = frns2select(np.array([41, 42, 43, 44, 45]) + 10 + 75 * Mode, LEDPeriod, 14, TotalFrames)
                dsout_Period = 5
            elif TotalFrames == 572:  # Cluster (13) LEDs
                # Pressure jumps after 156, and 429
                Nmin = 156 + 20
                Nmax = 429
                Count = np.floor((Nmin - Nmax + 1) / LEDPeriod)
                dsout_Select = frns2select(np.arange(Nmin - 1, Nmin + LEDPeriod), LEDPeriod,
                                           int((Nmax - Nmin + 1) / LEDPeriod))
                dsout_Period = LEDPeriod
            elif TotalFrames == 988:
                Nmin = 156 + 20
                Nmax = 429
                Count = int((Nmin - Nmax + 1) / LEDPeriod)
                dsout_Select = frns2select(np.arange(Nmin - 1, Nmin + LEDPeriod), LEDPeriod,
                                           int((Nmax - Nmin + 1) / LEDPeriod))
                dsout_Period = LEDPeriod
            elif TotalFrames == 182:
                dsout_Select = np.arange(1, N * LEDPeriod + 1)
                dsout_Period = LEDPeriod
            elif TotalFrames == 320:
                dt = np.mean(diff(ds['MainTime']))
                tmp, Nmin = min(abs(ds['MainTime'] - (ds['MainTime'][140, :] + 2)))
                ii = np.arange(Nmin, Nmin + LEDPeriod)
                xx = ds['LEDLocsXY'][ii, 1] / 127
                xx = xx - round(xx)
                if ds['ProcessingModeds']['Wave'] == 2:
                    jj = np.where(xx > 0)
                else:
                    jj = np.where(xx <= 0)
                dsout_Select = frns2select(ii[jj], LEDPeriod, int(30 / (LEDPeriod * dt)))
                dsout_Period = len(jj)
            elif TotalFrames == 480:
                dt = np.mean(diff(ds['MainTime']))
                tmp, Nmin = min(abs(ds['MainTime'] - (ds['MainTime'][140, :] + 2)))
                ii = np.arange(Nmin, Nmin + LEDPeriod)
                xx = ds['LEDLocsXY'][ii, 1] / 127
                xx = xx - round(xx)
                if ds['ProcessingModeds']['Wave'] == 2:
                    jj = np.where(xx > 0)
                else:
                    jj = np.where(xx <= 0)
                dsout_Select = frns2select(ii[jj], LEDPeriod, int(30 / (LEDPeriod * dt)))
                dsout_Period = len(jj)
            else:
                strarr = {'LED sequence and/or protocol not supported!'}
    # 到这选帧都是正确的
    # if ds['ProcessingModeds']['Seq']==1:
    #     dsout_Select=np.arange(1,LEDPeriod*N+1)
    #     dsout_Period=LEDPeriod
    V = ds['ProcessingModeds']['LEDID']
    # .any() != -1
    if type(V) != int:
        dld = ds['LEDIDs']
        dselect = dsout_Select[0:dsout_Period]
        ids = ds['LEDIDs'][dsout_Select[0:dsout_Period] - 1]
        ids = ids.reshape(-1)
        V = V.reshape(-1)
        A = np.abs(np.tile(V, (len(ids), 1)).T - np.tile(ids, (len(V), 1)))
        ii = np.where(np.sum(A == 0, axis=0) > 0)[0]  # Indexes of LED IDs mentioned.
        iii = frns2select(ii, dsout_Period, len(dsout_Select) // dsout_Period)
        dsout_Select = dsout_Select[iii]
        dsout_Period = len(ii)

    if dsout_Period < 2:
        print('Error: Less than 2 LEDs detected!')  # 后面没写
    fds.update({'Select': dsout_Select})
    fds.update({'Period': dsout_Period})
    return fds, strarr