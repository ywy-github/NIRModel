import numpy as np
# import distmx
# import ledcoords
def addhw(ds):
    TotalFrames, Sy, Sx = ds['Main_image'].shape#到这也是一致的

    NDarks, tmp1, tmp2= ds['Dark_image'].shape
    if NDarks == 2:
        ds['DarkDiff'] = ds['Dark_image'][1,:, :].astype(np.float32) - ds['Dark_image'][0,:, :].astype(np.float32)
        ds['Dark_image'] = 0.5 * (ds['Dark_image'][1,:, :].astype(np.float32) + ds['Dark_image'][0,:, :]).astype(np.float32)
        # ds.DarkStat = ds.DarkStat[0]
    else:
        ds['DarkDiff'] = np.zeros((Sy, Sx))
        ds['Dark_image'] = ds['Dark_image'][0,:, :]  # Load only the first dark frame
    #
    LevA = ds['ILState'][4, :].astype(float)
    LevB = ds['ILState'][5, :].astype(float)
    LevPWM = ds['ILState'][9, :].astype(float)#目前到这数值是一样的
    # if not np.array_equal(LevA, LevB):
    #     raise ValueError('LED Error: Levels A and B must be equal!')
    di=ds['ILState']
    ds['LEDIDs'] = ds['ILState'][10, :] + 1#到这也是一致的
    dl=ds['LEDIDs']
    # dsledids=ds['LEDIDs']
    if Sy == 64 and Sx == 128:
        ds.HWVersion = 1
        ds.LevLED = LevA * LevPWM / (255 * 2 ** 13)
        ds.CmPerPixel = 0.17
        ds.LEDLocsXY = np.repeat([[-1, 0], [0, 0], [1, 0]], TotalFrames // 3, axis=0)
        ds.LEDPeriod = 3
    elif (Sy == 102 and Sx == 128) or (Sy == 51 and Sx == 64) and np.sum(np.abs(LevPWM)) == 0:
        ds['HWVersion'] = 2
        FocIllumRelCoef = 2.0
        ds['LevLED'] = FocIllumRelCoef * LevA / 2 ** 13#这是一样的
        # dslev=ds['LevLED']
        ds['CmPerPixel'] = 0.19 if Sy == 102 else 2 * 0.19#这是一样的
        # dscm=ds['CmPerPixel']
        # if ds.ProcessingModeds.LittleLED == 2:
        #     if 'WAVE2' in ds.ProcessingModeds and ds.ProcessingModeds.WAVE2 == 1:
                # ds.LevLED = FocIllumRelCoef * LevA * (65535 / 5300) / 2 ** 13
        LEDPeriod = 1
        if TotalFrames > 1:
            Ind = np.where(ds['LEDIDs'][1:] == ds['LEDIDs'][0])[0]+1
            if Ind.any():
                LEDPeriod = np.min(Ind)
            else:
                LEDPeriod=TotalFrames
        # N = TotalFrames // LEDPeriod
        # Mx = np.reshape(ds['LEDIDs'][:N * LEDPeriod], (LEDPeriod, N))
        # A = Mx - np.tile(Mx[:, 0], (1, N))
        # if np.count_nonzero(A) != 0:
        #     print(Mx)
        #     raise ValueError('Error: Inconsistent LED IDs!')
        # if TotalFrames in [320, 480] and ds.HWVersion != 2:
        #     # ledlocs3 = ledtable3()
        #     V = ledlocs3[:, 0]
        #     ledlocs3 = ledlocs3[:, 2:4]
        #     Vtmp, ii = np.min(distmx(V, ds.LEDIDs))
        #     if np.max(Vtmp) > 0.001:
        #         raise ValueError('LED ID not found!')
        #     LEDLocsXY = ledlocs3[ii]
        # else:
        #     # ledlocs2 = ledcoords(2)
        #     LEDLocsXY = ledlocs2[ds.LEDIDs - 1]
        # print('LEDxx:', LEDLocsXY[:, 0].tolist())
        # print('LEDyy:', LEDLocsXY[:, 1].tolist())
        # ds.LEDLocsXY = 1.27 * LEDLocsXY
        # if ds['ProcessingModeds']['LittleLED'] != 0:
        #     inds = np.where(np.logical_or(ds['LEDIDs'] == 7, ds['LEDIDs'] == 9))[0]
        #     if len(inds) > 0:
        #         ds['LEDLocsXY'][inds] = ds['LEDLocsXY'][inds] / 1.27 * 0.537
        #     inds = np.where(ds.LEDIDs < 7)[0]
        #     if len(inds) > 0:
        #         ds['LEDLocsXY'][inds] = ds['LEDLocsXY'][inds] + 1.27
        #     inds = np.where(np.logical_and(ds['LEDIDs'] > 9, ds['LEDIDs'] <= 15))[0]
        #     if len(inds) > 0:
        #         ds['LEDLocsXY'][inds] = ds['LEDLocsXY'][inds] - 1.27
        ds['LEDPeriod'] = LEDPeriod
        # LEDPeriod = TotalFrames if TotalFrames == 1 else np.min(np.where(ds['LEDIDs'][1:] == ds['LEDIDs'][0]) + 1)
        # if TotalFrames==1:
        #     LEDPeriod =1
        # else:
        #     Ind=np.where(ds['ILState'][10, 1:] == ds['ILState'][10, 0])[0]-1
        #     if any(Ind):
        #         LEDPeriod=min(Ind)
        #     else:
        #         LEDPeriod =TotalFrames
        # N = TotalFrames // LEDPeriod
        # Mx = np.reshape(ds.LEDIDs[:N * LEDPeriod], (LEDPeriod, N))
        # A = Mx - np.tile(Mx[:, 0], (N, 1)).T
        # if np.any(A != 0):
        #     print(Mx)
        #     raise ValueError('Error: Inconsistent LED IDs!')

    #     if TotalFrames in [320, 480] and ds.HWVersion != 2:
    #         from ledtable3 import ledtable3
    #         V = ledtable3[:, 0]
    #         ledlocs3 = ledtable3[:, 2:4]
    #         Vtmp, ii = np.min(distmx(V, ds.LEDIDs.flatten())), np.argmin(distmx(V, ds.LEDIDs.flatten()))
    #         if np.max(Vtmp) > 0.001:
    #             raise ValueError('LED ID not found!')
    #         LEDLocsXY = ledlocs3[ii, :]
    #     else:
    #         from ledcoords import ledcoords
    #         ledlocs2 = ledcoords(2)
    #         LEDLocsXY = ledlocs2[ds.LEDIDs - 1, :]
    #
    #     print('LEDxx:', LEDLocsXY[:LEDPeriod, 0].tolist())
    #     print('LEDyy:', LEDLocsXY[:LEDPeriod, 1].tolist())
    #     ds.LEDLocsXY = 1.27 * LEDLocsXY
    #
    #     if ds.ProcessingModeds.LittleLED != 0:
    #         inds = np.where((ds.LEDIDs == 7) | (ds.LEDIDs == 9))[0]
    #         if len(inds) > 0:
    #             ds.LEDLocsXY[inds] = ds.LEDLocsXY[inds] / 1.27 * 0.537
    #         inds = np.where(ds.LEDIDs < 7)[0]
    #         if len(inds) > 0:
    #             ds.LEDLocsXY[inds] = ds.LEDLocsXY[inds] + 1.27
    #         inds = np.where((ds.LEDIDs > 9) & (ds.LEDIDs <= 15))[0]
    #         if len(inds) > 0:
    #             ds.LEDLocsXY[inds] = ds.LEDLocsXY[inds] - 1.27
    #
    #     ds.LEDPeriod = LEDPeriod
    # else:
    #     raise ValueError('File Error: HW Version is not supported!')

    dsout = ds
    return dsout
