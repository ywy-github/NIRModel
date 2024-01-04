from dnir_image.tool_code.matlab_python.comfortproc import comfortproc
from dnir_image.tool_code.matlab_python.file2ds import Read_info_from_dcm
import scipy.io as sio
from matplotlib import pyplot as plt
file_dcm= '../../../guoyang_images/2204/0571-ZKYZL-S219-CAJI-202301041457-双波段15-L.dcm'
ds = Read_info_from_dcm(file_dcm)
ds, strarr = comfortproc(ds)
sio.savemat('file.mat', {'data': ds})