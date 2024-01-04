import copy
def selectdata(ds, ind):
    das=copy.deepcopy(ds)#这里注意的是dict为可变对象因此必须创建副本
    # Select data for different waves
    ss1=das['Main_image']
    das['Main_image'] = das['Main_image'][ind,:, :]
    das['Main_stat'] = das['Main_stat'][ind]
    das['MainTime'] = das['MainTime'][ind]
    das['MainPressure'] = das['MainPressure'][ind]
    das['MainPressionTime'] = das['MainPressionTime'][ind]
    das['ILState'] = das['ILState'][:,ind]
    # ss1,ds2,ds3,ds4,ds5,ds6=ds['Main_image'],ds['Main_stat'],ds['MainTime'],ds['MainPressure'],ds['MainPressionTime'],ds['ILState']
    #到这都是一样的
    return das
