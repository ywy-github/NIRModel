import numpy as np
import copy
import scipy.io as sio

from data_preprocess.tool_code.matlab_python.ds2procds import ds2procds
from data_preprocess.tool_code.matlab_python.ds2procmode import ds2procmode
from data_preprocess.tool_code.matlab_python.selectdata import selectdata


# from file2ds import Read_info_from_dcm
def comfortproc(ds):
    dpp=ds['ProcessingModeds']
    dp=copy.deepcopy(ds)
    flag = 1
    dsout = {}
    si=ds['Main_image']#这也是对的上的
    # t = time.process_time()
    dsout['ErrorStatus'] = 0  # No Error
    dsout['ErrorString'] = ''  # Error String
    dsout['WarningStringArray'] = []
    # Check Verification Flag:
    VeriFlag = 0
    if 'Verification' in ds:
        VeriFlag = ds['Verification']#这也是对的上的

    if 'DataType' not in ds:
        ds['DataType'] = 0#这行也对的上
    # Experimental Processing Modes:
    pmds, strarr = ds2procmode(ds)
    ds['ProcessingModeds'] = pmds
    if 'NIRImage' not in dp['ProcessingModeds']:
        ds['ProcessingModeds'].update({"NIRImage":'0'})
    else:
        ds['ProcessingModeds'].update({'NIRImage':dp['ProcessingModeds']['NIRImage']})
    if 'LittleLED' not in dp['ProcessingModeds']:
        ds['ProcessingModeds'].update({"LittleLED": '0'})
    else:
        ds['ProcessingModeds'].update({'LittleLED':dp['ProcessingModeds']['LittleLED']})
    if 'TwoLEDMode' not in dp['ProcessingModeds']:
        ds['ProcessingModeds'].update({"TwoLEDMode": ''})
    else:
        ds['ProcessingModeds'].update({'TwoLEDMode': dp['ProcessingModeds']['TwoLEDMode']})
    dbefor=ds['ProcessingModeds']
    if strarr:
        dsout['WarningStringArray'] += strarr
        dsout['ErrorStatus'] = 2
    c=ds['MainTime'][1] - ds['MainTime'][0]
    # Add one wave 700ms Scan mode for frames selecting
    if ds['MainTime'][1] - ds['MainTime'][0] > 0.6:#到这是对的上的
        ds['ProcessingModeds']['PressureTimeC'] = 2

    # Split wave data for two wave dcm
    TotalFrames = ds['Main_image'].shape[0]#这也是一样的
    dstate=ds['ILState']#这也是对的
    # idx2=ds['ILState'][10, 1:]
    idx=np.where(ds['ILState'][10, 1:] == ds['ILState'][10, 0])[0]
    Ind = np.where(ds['ILState'][10, 1:] == ds['ILState'][10, 0])[0]+1

    Ind=np.transpose(Ind)#这块数值一致，但是行列不一致
    if not Ind.any():
        LEDPeriod = TotalFrames
    else:
        LEDPeriod = np.min(Ind)
    # Select 2LED ID
    if '1' in ds['ProcessingModeds']['TwoLEDMode']:
        ds['ProcessingModeds']['LEDID'] = np.array([ds['ILState'][10, 0], ds['ILState'][10, LEDPeriod-1]]) + 1

    if ds['ProcessingModeds']['WaveBandCount'] == 2:
        ds['ProcessingModeds']['twowaveDcm'] = flag
        # Only wave1: twowaveDcm = 1; dcm sub: twowaveDcm = 2;
        wave1Ind = np.where(ds['ILState'][7, :] == 1)[0]#索引和matlab差1
        wave2Ind = np.where(ds['ILState'][7, :] == 2)[0]
        ds2 = selectdata(ds, wave2Ind)
        ds = selectdata(ds, wave1Ind)

    # Prepare Data
    ds, strarr = ds2procds(ds)
    ddm=ds['Main_image']
    # datafram = ds['ProcessingModeds']["NIRImage"]
    # fds=ds2procds(ds)#输出LEDPerid,以及选帧
    # Judge mode for two wave dcm process mode
    if 'twowaveDcm' in ds['ProcessingModeds']:
        # if ds['ProcessingModeds']['twowaveDcm'] != 1:
            # Pre-process for wave 2
        ds2['ProcessingModeds']['WAVE2'] = 1
        ds2, strarr2 = ds2procds(ds2)
            # print(ds2)
            #### 0730
        return ds, ds2, strarr, strarr2
    #
    # if strarr:
    #     dsout['WarningStringArray'] += strarr
    #     dsout['ErrorStatus'] = 2

    # Convert Data to Matrix Format
    # if 'twowaveDcm' in ds['ProcessingModeds']:
    #     if ds['ProcessingModeds']['twowaveDcm'] in [2, 4, 5, 8, 9]:
    #         # Pre-process for wave 2
    #         ds = ds2mxds_TwoWaveSub(ds, ds2)
    #     elif ds['ProcessingModeds']['twowaveDcm'] == 3:
    #         # Wave1 sub wave3
    #         ds = ds2mxds_SubBeforeGRecon(ds, ds2)
    #     elif ds['ProcessingModeds']['twowaveDcm'] == 6:
    #         # PCA fusion
    #         ds = ds2mxds_PCA(ds, ds2)
    #     elif ds['ProcessingModeds']['twowaveDcm'] == 7:
    #         # NSST fusion
    #         ds = ds2mxds_NSST(ds, ds2)
    #     else:
    #         ds = ds2mxds(ds)
    # else:
    #     ds = ds2mxds(ds)

    # Pass Stuff to Output Data Structure
    # dsout['Params'] = np.zeros((1, 5))
    # dsout['CmPerPixel'] = ds['CmPerPixel']
    # dsout['TimePerCycle'] = ds['TimePerCycle']
    # dsout['Contour'] = ds['Contour']
    # dsout['Pressure'] = ds['Pressure']
    # Np = len(dsout['Pressure'])
    # dsout['PressureTime'] = (ds['DataSize'][2] - 1) * float(dsout['TimePerCycle']) * np.arange(Np) / (Np - 1)

    # Scan Quality Check
    # StrArr, dsout['SaturatedPixInds'] = qccheck(ds)
    # if StrArr:
    #     dsout['WarningStringArray'] += StrArr
    #     dsout['ErrorStatus'] = 2

    # return dsout
    return ds, [], strarr, []
# file_dcm=r'D:\DOBI 选帧程序Matlab转Python\DIcom\两小灯模式\3LED\KERZL6H1.dcm'
# file_dcm=r'D:\DOBI 选帧程序Matlab转Python\DIcom\双波段_3_5LED\5LED\020-ZSSYX-00314-HQHA-202203251011-双波段-L-D.dcm'
#file_dcm=r'D:\DOBI 选帧程序Matlab转Python\DIcom\双波段_3_5LED\3LED\020-ZSSYX-00261-ZAAN-202203180954-双波段-L-D.dcm'
#file_dcm=r'D:\DOBI 选帧程序Matlab转Python\DIcom\单波段_3_5_LED\5LED\C20006A_0004772-王丽凤-202303030827-D-L.dcm'
#file_dcm = r'D:\DOBI 选帧程序Matlab转Python\DIcom\单波段_3_5_LED\3LED\C20006A_0004773-卢大英-202303030837-D-R.dcm'
#file_dcm=r'D:\DOBI 选帧程序Matlab转Python\DIcom\700ms_3_5LED\3LED\J8X5F7H1.dcm'#这个LEDPeriod=5
#file_dcm=r'D:\DOBI 选帧程序Matlab转Python\DIcom\红外_双_3_5_LED（无单波段模式）\3LED\0571-ZKYZL-S200-LSME-202211091349-双波段15-R.dcm'
# 以上dicom全部测试成功
# ds = Read_info_from_dcm(file_dcm)
# ds, strarr = comfortproc(ds)
# sio.savemat('file.mat', {'data': ds})