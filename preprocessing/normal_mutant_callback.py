import numpy as np

from utils.data_distribution_manager import DataDistribution
from utils.mutant_callback import MutantCallback


class NormalMutantCallback(MutantCallback):
    def __init__(self, distribution: DataDistribution):
        self.distribution = distribution

    def mutant_data(self, data):
        mean, std = self.distribution.get_distribution(data)

        new_data = []
        for i in range(6):
            new_data.append(np.random.normal(loc=mean[i], scale=std[i]))

        return np.array(new_data)
