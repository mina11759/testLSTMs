import time
import argparse
import re

from preprocessing.PRSA_distribution import PRSAdistribution
from preprocessing.PRSA_data import PRSAData
from model.temperature import Temperature
from test_nn import *
from preprocessing.normal_mutant_callback import NormalMutantCallback
from preprocessing.oracle_einsum import OracleEinsum
# normalmutantcallback
# temperaturedistribution

def main():
    parser = argparse.ArgumentParser(description='testing for LSTM')
    parser.add_argument('--model', dest='model_name', default='PRSA', help='')
    parser.add_argument('--seed', dest='seed_num', default='2000', help='')
    parser.add_argument('--threshold_cc', dest='threshold_cc', default='5', help='')
    parser.add_argument('--threshold_gc', dest='threshold_gc', default='0.705', help='')
    parser.add_argument('--symbols_sq', dest='symbols_sq', default=2, help='')
    parser.add_argument('--seq', dest='seq', default='[7,11]', help='')

    args = parser.parse_args()

    model_name = args.model_name
    seed = int(args.seed_num)
    threshold_cc = float(args.threshold_cc)
    threshold_gc = float(args.threshold_gc)
    symbols_sq = int(args.symbols_sq)
    seq = args.seq
    seq = re.findall(r"\d+\.?\d*", seq)

    # radius = 0.005 # <<change?>>
    # test = None

    radius = 0.6

    if 'M2O_Stacked' in model_name:
        data_distribution = PRSAdistribution()
        mutant_callback = NormalMutantCallback(data_distribution)
        oracle = OracleEinsum(radius)
        data_manager = PRSAData(mutant_callback, oracle)
        model_manager = Temperature(model_name)
        test = TestNN(data_manager, model_manager, seed)

        test.lstm_test(threshold_cc, threshold_gc, symbols_sq, seq)

    elif 'M2O_Multilayered' in model_name:
        data_distribution = PRSAdistribution()
        mutant_callback = NormalMutantCallback(data_distribution)
        oracle = OracleEinsum(radius)
        data_manager = PRSAData(mutant_callback, oracle)
        model_manager = Temperature(model_name)
        test = TestNN(data_manager, model_manager, seed)

        test.lstm_test(threshold_cc, threshold_gc, symbols_sq, seq)

    elif 'M2M_Multilayered' in model_name:
        data_distribution = PRSAdistribution()
        mutant_callback = NormalMutantCallback(data_distribution)
        oracle = OracleEinsum(radius)
        data_manager = PRSAData(mutant_callback, oracle)
        model_manager = Temperature(model_name)
        test = TestNN(data_manager, model_manager, seed)

        test.lstm_test(threshold_cc, threshold_gc, symbols_sq, seq)

    else:
        data_distribution = PRSAdistribution()
        mutant_callback = NormalMutantCallback(data_distribution)
        oracle = OracleEinsum(radius)
        data_manager = PRSAData(mutant_callback, oracle)
        model_manager = Temperature(model_name)
        test = TestNN(data_manager, model_manager, seed)

        test.lstm_test(threshold_cc, threshold_gc, symbols_sq, seq)

if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))