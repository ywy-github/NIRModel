import pydicom
import numpy as np
import os
import scipy.io as si
import glob
import math
# def fileparts(file_dcm):
#     ext='.dcm'#重写
#     FName=file_dcm
#     pathstr=os.path.dirname(file_dcm)
#     name=os.path.basename(file_dcm)[:-4]
#     return pathstr,name,ext,FName
# pathstr,name,ext,FName=fileparts(file_dcm)
# def strcmp(str1,str2):
#     if str1[:len(str2)] == str2:
#         return 1
#     elif str2[:len(str1)] == str1:
#         return 1
#     return 0
# if strcmp(ext,'.mat'):
#     ds=si.loadmat(FName)
#     ds=ds.RawDataStructure
# elif strcmp(ext,'.dcm'):

def uncoding_mark(byte):
    if byte is None:
        return byte  # 如果 byte 是 None，直接返回，避免出现 TypeError
    if type(byte) == type(1):
        return byte
    i = 0
    numlist = []
    while i < len(byte):
        a = byte[i + 1]
        b = byte[i]
        num = a * 256 + b
        i += 2
        numlist.append(num)
    return np.array(numlist)

def Read_info_from_dcm(DcmFileName): # matlab file2ds

    Data = pydicom.dcmread(DcmFileName)
    answer = {}
    image_matrix = Data.pixel_array
    length = image_matrix.shape[0]
    width=image_matrix.shape[2]
    totalframes=image_matrix.shape[0]
    Dark_image = image_matrix[length - 2:length]
    Main_image = image_matrix[:length - 2]#减了两针
    Main_stat = np.zeros([1, length-2])
    Dark_stat = np.zeros([1, 2])
    CurveData_0 = Data.get_item([0x5000, 0x3000]).value # 对应 Matlab CurveData_0
    CurveData_0 = uncoding_mark(CurveData_0)
    CurveData_2 = Data.get_item([0x5002, 0x3000]).value # 对应 Matlab CurveData_2
    CurveData_2 = uncoding_mark(CurveData_2)#这是一样的
    CurveLabel = Data.get_item([0x5000, 0x2500]).value
    if CurveLabel == "DOBI IMAGE 2":
        DICOMRev = 2
        Step = 20
        dn = 1
    elif CurveLabel == "DOBI IMAGE 1":
        DICOMRev = 1
        Step = 19
        dn = 0
    DICOMRev = 2
    Step = 20
    dn = 1
    # else:
    #     print("error")
    #     exit(1)

    NFr = len(CurveData_2) / Step
    arrayinfo = CurveData_2.reshape(int(Step),int(NFr),order='F')
    arrayinfo = np.transpose(arrayinfo, (1, 0))
    arrayinfo = np.transpose(arrayinfo, (1, 0))
    # Main Time:
    if DICOMRev == 1:
        MainTime = 0.001 * CurveData_0[0, ..., 2]
    elif DICOMRev == 2:
        MainTime = 0.001*(2 ^ 16 * arrayinfo[0, :] + arrayinfo[1, :]).T
        # MainTime= 0.001 * (2 ** 16 * arrayinfo[0, :] + arrayinfo[1, :]).T
    # Main Stat:
    if DICOMRev == 2:
        A = CurveData_0.reshape(4, int(NFr),order='F') ####?
        A = np.transpose(A, (1, 0))
        Main_stat = A[:,3]#目前这个对的上
    MainPressure = arrayinfo[dn + 17,:]#对的上
    MainPressureTime = MainTime
    ILState = arrayinfo[dn + 5:dn + 16,:]#对的上
    Marks = Data.get_item([0x5003, 0x3000]).value
    Marks = uncoding_mark(Marks)

    if Marks is not None:
        MarkerP1 = [float(Marks[0] * 256 + Marks[1]), float(Marks[2] * 256 + Marks[3])]#这是一样的
        MarkerP2 = [float(Marks[4] * 256 + Marks[5]), float(Marks[6] * 256 + Marks[7])]
    else:
        # 处理 Marks 为 None 或长度小于 4 的情况
        MarkerP1 = [0.0, 0.0]  # 或者其他适当的处理
        MarkerP2 = [0.0, 0.0]
    Laterality = Data.get_item([0x0020, 0x0060]).value
    flush = Data.get_item([0x0028, 0x0008]).value
    Rows = Data.get_item([0x0028, 0x0010]).value
    Rows = uncoding_mark(Rows)
    Columns = Data.get_item([0x0028, 0x0011]).value
    Columns = uncoding_mark(Columns)
    NIRImage = Data.get_item([0x0091, 0x1002])
    # XXQ NIR version
    ProcessingMode={}
    ##没有合适的dicom，这块待验证
    if Data.get_item([0x0053, 0x1001]) is not None:
        # ProcessingMode+='LittleLED=' + str(Data.get_item([0x0053, 0x1001]).value)
        ProcessingMode.update({"LittleLED":str(uncoding_mark(Data.get_item([0x0053, 0x1001]).value))})#字典形式写法
    if Data.get_item([0x0091, 0x1002]) is not None:
        # ProcessingMode+='NIRImage='+str(Data.get_item([0x0091, 0x1002]).value)
        ProcessingMode.update({"NIRImage": str(uncoding_mark(Data.get_item([0x0091, 0x1002]).value))})
    if Data.get_item([0x0053,0x1002]) is not None:
        ProcessingMode.update({"TwoLEDMode=":str(uncoding_mark(Data.get_item([0x0053, 0x1002]).value))})
    if NIRImage is None:
        NIRImage = 0
    else:
        NIRImage = NIRImage.value
    # answer.update({"NIRImage": NIRImage})
    answer.update({"DICOMRev": DICOMRev})
    answer.update({"Laterality": Laterality})
    answer.update({"flush": flush})
    answer.update({"Rows": int(Rows)})
    answer.update({"Columns": int(Columns)})
    answer.update({"MainPressionTime": MainTime})
    answer.update({"Main_stat": Main_stat})
    answer.update({"MainPressure": MainPressure})
    answer.update({"ILState": ILState})
    answer.update({"MarkerP1": MarkerP1})
    answer.update({"MarkerP2": MarkerP2})
    # answer.update({"ProcessingMode":ProcessingMode})
    answer.update({"length":length})
    answer.update({"totalframes":totalframes})
    answer.update({"width":width})
    answer.update({"Main_image":Main_image})
    answer.update({"MainTime":MainTime})
    answer.update({"LEDPeriod":0})
    answer.update({"Dark_image":Dark_image})
    answer.update({"ProcessingModeds":ProcessingMode})
    # ProcessingMode.update({'NIRImage':NIRImage})
    return answer
# Read_info_from_dcm(file_dcm)
