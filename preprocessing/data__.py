# """
#     Preprocessing by time series
# """

import numpy as np
import csv

# with open('PRSA_data_2010.1.1-2014.12.31.csv', newline='') as csvfile:
#     data = []
#     reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
#     for row in reader:
#         element = ', '.join(row)
#         # print(', '.join(row))
#         data.append(element)


def PRSA_Preprocessing():
    f = open('src/model/PRSA_data_2010.1.1-2014.12.31.csv', 'r')
    rdr = csv.reader(f)
    PRSA_data = []
    for line in rdr:
        # print(line[5:])
        PRSA_data.append(line[5:])
    f.close()

    PRSA_data = PRSA_data[1:][:]
    # print(np.shape(PRSA_data))

    n_data = np.size(PRSA_data, 0) # 43824
    n_feature = np.size(PRSA_data, 1) # 6

    return PRSA_data, n_data, n_feature


