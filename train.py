import time

# from data.oracle_einsum import OracleEinsum
from preprocessing.PRSA_data import PRSAData
from preprocessing.PRSA_distribution import PRSAdistribution
# from data.temperature.normal_mutant_callback import NormalMutantCallback
from model.temperature import Temperature

# model_names = ['temperature_no_norm', 'temperature_no_norm_first_batch', 'temperature_no_norm_third_batch',
#                'temperature_no_norm_six_batch', 'temperature_no_norm_all_batch']
# model_names = ['temperature_no_norm_six_batch', 'temperature_no_norm_all_batch']
model_names = ['M2O_Stacked', 'M2O_Multilayered', 'M2M_Stacked', 'M2M_Multilayered']

results = []
times = []

for model_name in model_names:

    radius = 0.6

    data_distribution = PRSAdistribution()
    # mutant_callback = NormalMutantCallback(data_distribution)
    # oracle = OracleEinsum(radius)
    data_manager = PRSAData()
    model_manager = Temperature(model_name)

    (x_train, y_train) = data_manager.get_train_data()
    (x_test, y_test) = data_manager.get_test_data()
    # for i in range(10):
    if 'M2O_Stacked' in model_names:
        model_manager.train_model_M2O_Stacked(x_train, y_train, x_test, y_test)
    elif 'M2O_Multilayered' in model_names:
        model_manager.train_model_M2O_Multilayered(x_train, y_train, x_test, y_test)
    elif 'M2M_Stacked' in model_names:
        model_manager.train_model_M2M_Stacked(x_train, y_train, x_test, y_test)
    else:
        model_manager.train_model_M2M_Multilayered(x_train, y_train, x_test, y_test)

    start_time = time.time()

    results.append(model_manager.test_model(data_manager.x_test, data_manager.y_test))
    model_manager.test_model(data_manager.x_train, data_manager.y_train)

    end_time = time.time() - start_time
    times.append(end_time)
    print("--- %s seconds ---" % end_time)
    print(len(times))

# f = open('output/result.txt', 'w')
# for i in range(len(model_names)):
#     for j in range(4):
#         f.write('%d  %s time: %f\n' % (j, model_names[i], times[i * 10 + j]))
#         f.write('%d  %s result: %f\n' % (j, model_names[i], results[i * 10 + j]))
# f.close()



