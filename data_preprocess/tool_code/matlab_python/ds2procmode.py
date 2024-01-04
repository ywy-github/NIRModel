import copy
import pydicom
def ds2procmode(ds):
    # Processing String Parameters
    pmds = {
        'BulkA': 0,
        'BulkD': 3,
        'Butfl': 6,
        'Cone': 0,
        'Contour': 1,
        'CPFilt': 0,
        'CPlane': 2,
        'DetMode': 0,
        'DefOrder': 1,
        'DiffMode': 0,
        'Float': 0,
        'FloatFilt': 0,
        'InfVein': 0,
        'Jacob': 1,
        'Jump': 0,
        'Layers': 0,
        'LEDID': -1,
        'Norm': 0,
        'NumLs': -1,
        'NVoxLs': 1,
        'OutFilt': 0,
        'Proj': 0,
        'Reg': -1,
        'RegAmp': 0,
        'RegAmpAdj': 0,
        'RegDet': 1,
        'RegDetAmp': 1,
        'RegLED': 2,
        'Regress': 0,
        'RegVoxSm': 0,
        'RegVoxSmL': 0,
        'Resp': 1,
        'SensComp': 0,
        'SensCone': 0,
        'Seq': 0,
        'Shape': 3,
        'SmoothVox': 0,
        'Spline': 0,
        'Static': 0,
        'Sup': -1,
        'T0': 0,
        'Test': 0,
        'TSmooth': 3,
        'Vein': 1,
        'VoxAmp': -1,
        'VoxAmpAuto': 0,
        'VoxDC': 0,
        'VoxFilt': 0,
        'VoxSVD': 0,
        'Wave': 1,
        'WaveBandCount': len(set(ds['ILState'][7])),
        'TwoLEDMode': 0
    }
    # Try to redefine Defaults:
    strarr = []  # Error/Warning string
    if 'ProcessingMode' in ds:
        try:
            strds = dict(item.split('=') for item in ds['ProcessingMode'].split(';'))
            # Cycle through entered fields:
            for key, value in strds.items():
                if key in pmds:
                    pmds[key] = int(value)  # Convert to integer
                else:
                    strarr.append(f"Parameter {key} not supported!")
        except Exception as e:
            strarr.append('Problem Decoding Processing Mode:')
            strarr.append(ds['ProcessingMode'])
            strarr.append(str(e))

    return pmds, strarr
