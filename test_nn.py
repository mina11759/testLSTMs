import random
import numpy as np
from tqdm import tqdm

from utils.data_manager import DataManager
from model.model_manager import ModelManager
from model.threshold_manager import ThresholdManager
from model.state_manager import StateManager

from coverage.cell_coverage import CellCoverage
from coverage.gate_coverage import GateCoverage
from coverage.negative_sequence_coverage import NegativeSequenceCoverage
from coverage.positive_sequence_coverage import PositiveSequenceCoverage


class TestNN:
    def __init__(self, data_manager: DataManager, model_manager: ModelManager, seed):
        self.data_manager = data_manager
        self.model_manager = model_manager
        self.model_manager.load_model()

        self.target_data = self.data_manager.x_train[random.sample(range(np.shape(self.data_manager.x_train)[0]), seed)] # 이걸 바꿔야할듯?

    def train(self):
        (x_train, y_train) = self.data_manager.get_train_data()
        (x_test, y_test) = self.data_manager.get_test_data()
        self.model_manager.train_model_M2O_Stacked(x_train, y_train, x_test, y_test)

    def data_process(self, coverage_set, target_data):
        pbar = tqdm(range(len(target_data)), total=len(target_data))
        for i in pbar:
            before_output = self.model_manager.get_prob(target_data[i]) # 예측하는 것
            generated_data, _ = self.data_manager.mutant_data(target_data[i])

            after_output = self.model_manager.get_prob(generated_data) # 얘도 예측하는 것
            self.data_manager.update_sample(before_output, after_output, target_data[i], generated_data)

            for coverage in coverage_set:
                if not (generated_data is None):
                    coverage.update_features(generated_data)
                    coverage.update_graph(self.data_manager.get_num_samples() / len(coverage_set))

        for coverage in coverage_set:
            coverage.save_feature()
            coverage.update_frequency_graph()
            coverage.display_graph()
            coverage.display_frequency_graph()
            coverage.display_stat()

        for coverage in coverage_set:
            _, result = coverage.calculate_coverage()
            print("%s : %.5f" % (coverage.get_name(), result))

    def lstm_test(self, threshold_cc, threshold_gc, symbols_sq, seq):
        self.model_manager.load_model()
        model = self.model_manager.model
        model_name = self.model_manager.model_name
        # _, other_layers = self.model_manager.get_lstm_layer()

        indices, lstm_layers = self.model_manager.get_lstm_layer()
        # for layer in lstm_layers:
        #     ThresholdManager(self.model_manager, layer, self.data_manager.x_train)
        init_data = self.target_data[15]
        print(init_data)
        layer = lstm_layers[-1]
        state_manager = StateManager(model, indices[-1])
        coverage_set = [CellCoverage(layer, model_name, state_manager, threshold_cc, init_data),
                        GateCoverage(layer, model_name, state_manager, threshold_gc, init_data),
                        PositiveSequenceCoverage(layer, model_name, state_manager, symbols_sq, seq),
                        NegativeSequenceCoverage(layer, model_name, state_manager, symbols_sq, seq)]

        self.data_process(coverage_set, self.target_data)