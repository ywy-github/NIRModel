import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

if __name__ == '__main__':
     file_path = "../data/ti_二期双十+双十五wave1原始图/train/benign/020-ZSSYX-0204-FXBI-202203100914-双波段-R-D.bmp"
     image = Image.open(file_path)
     img = np.array(image)
     print(img)