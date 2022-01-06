import numpy as np
import pandas as pd

from utils.data_distribution_manager import DataDistribution


class PRSAdistribution(DataDistribution):
    def __init__(self):
        self.means = []
        self.stds = []
        self.load_distribution()

    def load_distribution(self):
        train_data = pd.read_csv("./preprocessing/PRSA_data_2010.1.1-2014.12.31.csv", sep= ',')
        train_data = np.array(train_data)
        raw_data = train_data[:, 5:] # <<confirm>>

        sort_data = [[]] * 4

        for data in raw_data:
            sort_data[self.get_index(data)].append(data)

        model_data_by_col = []
        for data_list in sort_data:
            temp = []
            for col in range(6):
                temp_col = []
                for data in data_list:
                    temp_col.append(data[col])
                temp.append(temp_col)
            model_data_by_col.append(temp)
        model_data_by_col = np.array(model_data_by_col)

        means_by_file = []
        stds_by_file = []
        for file in range(4):
            means = []
            stds = []
            for mut_col in range(6):
                means.append(np.mean(model_data_by_col[file][mut_col]))
                stds.append(np.std(model_data_by_col[file][mut_col]))
            means_by_file.append(means)
            stds_by_file.append(stds)

        self.means = means_by_file
        self.stds = stds_by_file

        # print(means_by_file)
        # print(stds_by_file)
        # print(means)

    @staticmethod
    def get_index(data):
        temperature = data[-1]

        if 15 < temperature < 25:
            target_f_idx = 0
        elif temperature >= 25:
            target_f_idx = 1
        elif -5 < temperature < 15:
            target_f_idx = 2
        else:
            target_f_idx = 3

        return target_f_idx

    def get_distribution(self, data):
        f_idx = self.get_index(data)
        return self.means[f_idx], self.stds[f_idx]