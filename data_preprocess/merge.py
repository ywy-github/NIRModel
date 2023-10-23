import pandas as pd
import numpy as np

if __name__ == '__main__':
    data_2202 = pd.read_excel("F:\\PyCharm 2022.2.1\\pythonProject\\nir\\2202+932.xlsx")
    data_2204 = pd.read_excel("F:\\PyCharm 2022.2.1\\pythonProject\\nir\\2204+656.xlsx",sheet_name="2204例")

    dcm_name_2202=data_2202["dcm_name"]
    dcm_name_2204=data_2204["dcm_name"]
    difference = dcm_name_2204[~dcm_name_2204.isin(dcm_name_2202)]
    print(difference)
