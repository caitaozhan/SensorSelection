'''Testing
'''

from select_sensor import SelectSensor
try:
    from cuda_kernals import o_t_approx_kernal, o_t_kernal, o_t_approx_dist_kernal, \
                             o_t_approx_kernal2, o_t_approx_dist_kernal2, \
                             o_t_iter_kernal
except Exception as e:
    print(e)
import time
import plots
import numpy as np
import pandas as pd
import random
#import line_profiler
#from joblib import Parallel, delayed, dump, load


#Define constants for flags
BASELINE_ALL = 0
BASELINE_GA = 1
BASELINE_COV = 2
BASELINE_RAN = 3
BASELINE_AGA = 4
BASELINE_WAGA = 5
BASELINE_OPT = 6

LARGE_INSTANCE = 0
STD_INSTANCE = 1
SMALL_INSTANCE = 2
LARGE_INSTANCE_2 = 3

ONLY_INTRUDERS = 1
PRIMARY_INTRUDERS = 2
PRIMARY_SECONDARY_INTRUDERS = 3

RTL_SDR_NOISE_FLOOR = -80
WIFI_NOISE_FLOOR = -55

def test_ipsn_homo(algorithms=BASELINE_ALL):
    '''2019 IPSN version
    '''
    selectsensor = SelectSensor(grid_len=50)
    selectsensor.init_data('data50/homogeneous-200/cov', 'data50/homogeneous-200/sensors', 'data50/homogeneous-200/hypothesis')
    budget = 10
    # CPU version
    selectsensor.transmitters_to_array()
    #start = time.time()
    #selectsensor.select_offline_greedy_p_lazy_gpu(15, 12, o_t_approx_kernal)
    #print('time = {}'.format(time.time()-start))
    if algorithms == BASELINE_ALL or algorithms == BASELINE_AGA:
        # selectsensor.transmitters_to_array()
        results_AGA = selectsensor.select_offline_greedy_lazy_gpu(budget, 20, o_t_approx_kernal2)
        # print(results_AGA)
        # plots.save_data(results_AGA, 'plot_data_splat/fig1-homo/GA')

    if algorithms == BASELINE_ALL or algorithms == BASELINE_GA:  # GA
        # selectsensor.transmitters_to_array()
        results_GA = selectsensor.select_offline_GA(budget, o_t_iter_kernal)
        # plots.save_data(results_GA, 'plot_data_splat/fig1-homo/GA')

    if algorithms == BASELINE_ALL or algorithms == BASELINE_RAN:  # Random
        results_RAN = selectsensor.select_offline_random(budget, 20)
        # plots.save_data(results_RAN, 'plot_data_splat/fig1-homo/random')

    if algorithms == BASELINE_ALL or algorithms == BASELINE_COV:  # Coverage
        results_COV = selectsensor.select_offline_coverage(budget, 20)
        plots.save_data(results_COV, 'plot_data_splat/fig1-homo/coverage')

    print('')
    for i in range(0, budget + 1):
        print(i, end=' ')
        if algorithms == BASELINE_ALL or algorithms == BASELINE_AGA:
            print(results_AGA[i][2], end=' ')
        if algorithms == BASELINE_ALL or algorithms == BASELINE_GA:
            print(results_GA[i][1], end=' ')
        if algorithms == BASELINE_ALL or algorithms == BASELINE_RAN:
            print(results_RAN[i][1], end=' ')
        if algorithms == BASELINE_ALL or algorithms == BASELINE_COV:
            print(results_COV[i][1], end=' ')
        print('')

        #results = selectsensor.select_offline_greedy_lazy_gpu(50, budget, 20, o_t_approx_kernal2)
    #for r in results:
    #    print(r[:-1])

    #plots.figure_1a(selectsensor, None)

def test_weighted_ipsn(algorithms=BASELINE_ALL):
    grid_len = 50
    selectsensor = SelectSensor(grid_len=grid_len)
    selectsensor.init_data('data50/homogeneous-200/cov', 'data50/homogeneous-200/sensors',
                           'data50/homogeneous-200/hypothesis')
    budget = 10
    # CPU version
    selectsensor.transmitters_to_array()

    if algorithms == BASELINE_ALL or algorithms == BASELINE_AGA:
        # selectsensor.transmitters_to_array()
        results_AGA = selectsensor.select_offline_greedy_lazy_gpu(budget, 20, o_t_approx_kernal2)
        # print(results_AGA)
        # plots.save_data(results_AGA, 'plot_data_splat/fig1-homo/GA')

    if algorithms == BASELINE_ALL or algorithms == BASELINE_GA:  # GA
        # selectsensor.transmitters_to_array()
        results_GA = selectsensor.select_offline_GA(budget, o_t_iter_kernal)
        # plots.save_data(results_GA, 'plot_data_splat/fig1-homo/GA')

    if algorithms == BASELINE_ALL or algorithms == BASELINE_RAN:  # Random
        results_RAN = selectsensor.select_offline_random(budget, 20)
        # plots.save_data(results_RAN, 'plot_data_splat/fig1-homo/random')

    if algorithms == BASELINE_ALL or algorithms == BASELINE_COV:  # Coverage
        results_COV = selectsensor.select_offline_coverage(budget, 20)
        #plots.save_data(results_COV, 'plot_data_splat/fig1-homo/coverage')

    true_x = np.random.choice(range(grid_len), size=200, replace=True)
    true_y = np.random.choice(range(grid_len), size=200, replace=True)

    error_AGA = np.zeros(budget + 1)
    error_WAGA = np.zeros(budget + 1)
    error_GA = np.zeros(budget + 1)
    error_COV = np.zeros(budget + 1)
    error_RAN = np.zeros(budget + 1)
    for j in range(1, budget + 1):
        for tno, trans in enumerate(true_x):
            # if algorithms == BASELINE_ALL or algorithms == BASELINE_WAGA:
            #     print(results_WAGA[j - 1][3], selectsensor.means_rescale.shape, trans, true_y[tno])
            #     error_WAGA[j] += selectsensor.compute_conditional_error(trans, true_y[tno],
            #                                                             results_WAGA[j - 1][3])
            if algorithms == BASELINE_ALL or algorithms == BASELINE_AGA:
                error_AGA[j] += selectsensor.compute_conditional_error(trans, true_y[tno],
                                                                       results_AGA[j - 1][3])
                print('error_AGA = ', budget, error_AGA[j])
            if algorithms == BASELINE_ALL or algorithms == BASELINE_GA:
                error_GA[j] += selectsensor.compute_conditional_error(trans, true_y[tno],
                                                                      results_GA[j - 1][2])
            if algorithms == BASELINE_ALL or algorithms == BASELINE_COV:
                error_COV[j] += selectsensor.compute_conditional_error(trans, true_y[tno],
                                                                       results_COV[j - 1][2])
            if algorithms == BASELINE_ALL or algorithms == BASELINE_RAN:
                error_RAN[j] += selectsensor.compute_conditional_error(trans, true_y[tno],
                                                                       results_RAN[j - 1][2])
        error_AGA /= 100
        error_WAGA /= 100
        error_GA /= 100
        error_COV /= 100
        error_RAN /= 100
        # print(j, error_AGA[j], error_WAGA[j], error_GA[j], error_COV[j], error_RAN[j])
    print('')
    for j in range(budget + 1):
        print(j, end=' ')
        if algorithms == BASELINE_ALL or algorithms == BASELINE_AGA:
            #mean_AGA = np.mean(cumul_AGA[j][:])
            #std_AGA = np.std(cumul_AGA[j][:])
            print(error_AGA[j], end=' ')
        if algorithms == BASELINE_ALL or algorithms == BASELINE_GA:
            print(error_GA[j], end=' ')
        if algorithms == BASELINE_ALL or algorithms == BASELINE_RAN:
            print(error_RAN[j], end=' ')
        if algorithms == BASELINE_ALL or algorithms == BASELINE_COV:
            print(error_COV[j], end=' ')
        print('')


def gen_data(iteration, num_sensors=100, subdir = 'dataSplat/1600-400/', grid_len = 40, hetero=False):
    sensor_file = subdir + 'sensors' + str(iteration)
    cov_file = subdir + 'cov' + str(iteration)
    hypothesis_file = subdir + 'hypothesis' + str(iteration)

    all_locations = [(i, j) for i in range(grid_len) for j in range(grid_len)]

    sensors = random.sample(all_locations, num_sensors)
    print(sensors)
    ipsn = pd.read_csv('dataSplat/1600-400/sensors-scale-4', delimiter=' ', header=None)
    stds = ipsn[2]
    all_stds = np.unique(stds.values)
    if hetero:
        ipsn = pd.read_csv('dataSplat/1600-hetero/sensor_from_ipsn', delimiter=' ', header=None)
        std_costs = ipsn[3]
        std_costs = np.random.choice(std_costs, size=num_sensors, replace=True)
    else:
        std_costs = [1] * len(sensors) #Homogeneous costs

    with open(sensor_file, 'w') as f:
        for index, sensor in enumerate(sensors):
            # print(index, sensor)
            f.write('{} {} {} {}\n'.format(sensor[0], sensor[1], stds[index], std_costs[index]))
    f.close()
    with open(cov_file, 'w') as f:
        cov = np.zeros((num_sensors, num_sensors))
        for i in range(num_sensors):
            for j in range(num_sensors):
                if i == j:
                    cov[i, j] = std_costs[i] ** 2
                f.write('{} '.format(cov[i, j]))
            f.write('\n')
    f.close()

    hypo_template = subdir + 'tx_{:04d}_pathloss.txt'
    hypo_template2 = subdir + 'tx_{:d}_pathloss.txt'
    hypo = None
    hypothesis_file = open(hypothesis_file, 'w')
    for i in range(1, grid_len * grid_len + 1):
        if i % 10 == 0:
            print(i, end=' ')
        try:
            hypo = pd.read_csv(hypo_template.format(i), delimiter=' ', header=None)
        except:
            hypo = pd.read_csv(hypo_template2.format(i), delimiter=' ', header=None)
        #print(hypo)
        trans_x = (i - 1) // grid_len
        trans_y = (i - 1) % grid_len
        for index, sensor in enumerate(sensors):
            sen_x = sensor[0]
            sen_y = sensor[1]
            std = std_costs[index]
            mean = 30 - hypo.iloc[sen_y, sen_x]  # cellular tower power - pathloss
            print(trans_x, trans_y, sen_x, sen_y, mean, std, file=hypothesis_file)
    hypothesis_file.close()


def test_splat_baseline(size_instance, algorithms, num_iterations = 20):
    '''The baseline (GA, random, coverage), without background, homogeneous, 40 x 40 grid
    '''
    #random.seed(1)
    if size_instance is LARGE_INSTANCE:
        subdir = 'dataSplat/4096/'
        grid_len = 64
        budget = 40
        num_sensors = 400
    elif size_instance is STD_INSTANCE:
        subdir = 'dataSplat/1600-400/'
        grid_len = 40
        budget = 20
        num_sensors = 160
    elif size_instance is SMALL_INSTANCE:
        subdir = 'dataSplat/100/'
        grid_len = 10
        budget = 12
        num_sensors = 20
    elif size_instance is LARGE_INSTANCE_2:
        subdir = 'dataSplat/4096-2/'
        grid_len = 64
        budget = 30
        num_sensors = 400

    cov_file = subdir + 'cov'
    sensor_file = subdir + 'sensors'
    intruder_hypo_file = subdir + 'hypothesis'

    cumul_AGA = np.zeros((budget + 1, num_iterations))
    cumul_GA  = np.zeros((budget + 1, num_iterations))
    cumul_RAN = np.zeros((budget + 1, num_iterations))
    cumul_COV = np.zeros((budget + 1, num_iterations))
    cumul_OPT = np.zeros((budget + 1, num_iterations))
    for i in range(num_iterations):
        gen_data(i, num_sensors, subdir=subdir, grid_len=grid_len)
        selectsensor = SelectSensor(grid_len)
        cov_file_cur = cov_file + str(i)
        sensor_file_cur = sensor_file + str(i)
        intruder_hypo_file_cur = intruder_hypo_file + str(i)
        #cov_file_cur = cov_file
        #sensor_file_cur = sensor_file
        #intruder_hypo_file_cur = intruder_hypo_file
        selectsensor.init_data(cov_file_cur, sensor_file_cur, intruder_hypo_file_cur)
        selectsensor.rescale_intruder_hypothesis(noise_floor=-55)
        # print(selectsensor.means)
        selectsensor.transmitters_to_array()        # for GPU
        if algorithms == BASELINE_ALL or algorithms == BASELINE_AGA:
            #selectsensor.transmitters_to_array()
            results_AGA = selectsensor.select_offline_greedy_lazy_gpu(budget, 20, o_t_approx_kernal2)
            #print(results_AGA)
            #plots.save_data(results_AGA, 'plot_data_splat/fig1-homo/GA')

        if algorithms == BASELINE_ALL or algorithms == BASELINE_GA:  # GA
            #selectsensor.transmitters_to_array()
            results_GA = selectsensor.select_offline_GA(budget, o_t_iter_kernal)
            #plots.save_data(results_GA, 'plot_data_splat/fig1-homo/GA')

        if algorithms == BASELINE_ALL or algorithms == BASELINE_RAN:  # Random
            results_RAN = selectsensor.select_offline_random(budget, 20)
            #plots.save_data(results_RAN, 'plot_data_splat/fig1-homo/random')

        if algorithms == BASELINE_ALL or algorithms == BASELINE_COV:  # Coverage
            results_COV = selectsensor.select_offline_coverage(budget, 20)

        for j in range(1, len(results_AGA) + 1):
            if algorithms == BASELINE_ALL or algorithms == BASELINE_AGA:
                cumul_AGA[j][i] = results_AGA[j-1][2]
            if algorithms == BASELINE_ALL or algorithms == BASELINE_GA:
                cumul_GA[j][i] = results_GA[j-1][1]
            if algorithms == BASELINE_ALL or algorithms == BASELINE_RAN:
                #print(cumul_RAN, results_RAN[i][1])
                cumul_RAN[j][i] = results_RAN[j-1][1]
            if algorithms == BASELINE_ALL or algorithms == BASELINE_COV:
                cumul_COV[j][i] = results_COV[j-1][1]
    #print(cumul_AGA)
    # cumul_AGA  /= num_iterations
    # cumul_GA /= num_iterations
    # cumul_RAN /= num_iterations
    # cumul_COV /= num_iterations

    print('')
    for j in range(len(results_AGA) + 1):
        print(j, end = ' ')
        if algorithms == BASELINE_ALL or algorithms == BASELINE_AGA:
            mean_AGA = np.mean(cumul_AGA[j][:])
            std_AGA = np.std(cumul_AGA[j][:])
            print(mean_AGA, std_AGA, end = ' ')
        if algorithms == BASELINE_ALL or algorithms == BASELINE_GA:
            mean_GA = np.mean(cumul_GA[j][:])
            std_GA = np.std(cumul_GA[j][:])
            print(mean_GA, std_GA, end=' ')
        if algorithms == BASELINE_ALL or algorithms == BASELINE_RAN:
            mean_RAN = np.mean(cumul_RAN[j][:])
            std_RAN = np.std(cumul_RAN[j][:])
            print(mean_RAN, std_RAN, end=' ')
        if algorithms == BASELINE_ALL or algorithms == BASELINE_COV:
            mean_COV = np.mean(cumul_COV[j][:])
            std_COV = np.std(cumul_COV[j][:])
            print(mean_COV, std_COV, end=' ')
        if algorithms == BASELINE_ALL or algorithms == BASELINE_OPT:
            mean_OPT = np.mean(cumul_OPT[j][:])
            std_OPT = np.std(cumul_OPT[j][:])
            print(mean_OPT, std_OPT, end=' ')
        print('')

def test_varying_sensor_density(algorithms, num_iterations):
    num_sensor_list = [50, 100, 150, 200]
    budget_list = [5, 10, 15]
    results = np.zeros((len(num_sensor_list), len(budget_list), 8))
    for i, num_sensor in enumerate(num_sensor_list):
        list = test_varying_sensor_density_kernel(algorithms, num_iterations, num_sensor, budget_list[-1])
        for j, budget in enumerate(budget_list):
            results[i, j, :] = list[:, budget]

    for i, num_sensor in enumerate(num_sensor_list):
        for j, budget in enumerate(budget_list):
            print('>>>', num_sensor, budget, results[i][j][0],
                  results[i][j][1], results[i][j][2], results[i][j][3],
                  results[i][j][4], results[i][j][5], results[i][j][6],
                  results[i][j][7])

def test_varying_sensor_density_kernel(algorithms, num_iterations=1, num_sensors = 50, budget=5):
    subdir = 'dataSplat/1600-100/'
    cov_file = subdir + 'cov'
    sensor_file = subdir + 'sensors'
    intruder_hypo_file = subdir + 'hypothesis'
    # selectsensor.transmitters_to_array()  # for GPU
    grid_len = 40
    budget = 15

    cumul_AGA = np.zeros((budget + 1, num_iterations))
    cumul_GA = np.zeros((budget + 1, num_iterations))
    cumul_RAN = np.zeros((budget + 1, num_iterations))
    cumul_COV = np.zeros((budget + 1, num_iterations))
    for i in range(num_iterations):
        gen_data(i, num_sensors, subdir=subdir, grid_len=grid_len)
        selectsensor = SelectSensor(grid_len)
        cov_file_cur = cov_file + str(i)
        sensor_file_cur = sensor_file + str(i)
        intruder_hypo_file_cur = intruder_hypo_file + str(i)

        selectsensor.init_data(cov_file_cur, sensor_file_cur, intruder_hypo_file_cur)
        selectsensor.rescale_intruder_hypothesis()
        selectsensor.transmitters_to_array()  # for GPU
        if algorithms == BASELINE_ALL or algorithms == BASELINE_AGA:
            results_AGA = selectsensor.select_offline_greedy_lazy_gpu(budget, 20, o_t_approx_kernal2)

        if algorithms == BASELINE_ALL or algorithms == BASELINE_GA:  # GA
            results_GA = selectsensor.select_offline_GA(budget, o_t_iter_kernal)
        if algorithms == BASELINE_ALL or algorithms == BASELINE_RAN:  # Random
            results_RAN = selectsensor.select_offline_random(budget, 20)
        if algorithms == BASELINE_ALL or algorithms == BASELINE_COV:  # Coverage
            results_COV = selectsensor.select_offline_coverage(budget, 20)

        for j in range(1, len(results_AGA) + 1):
            if algorithms == BASELINE_ALL or algorithms == BASELINE_AGA:
                cumul_AGA[j][i] = results_AGA[j - 1][2]
            if algorithms == BASELINE_ALL or algorithms == BASELINE_GA:
                cumul_GA[j][i] = results_GA[j - 1][1]
            if algorithms == BASELINE_ALL or algorithms == BASELINE_RAN:
                # print(cumul_RAN, results_RAN[i][1])
                cumul_RAN[j][i] = results_RAN[j - 1][1]
            if algorithms == BASELINE_ALL or algorithms == BASELINE_COV:
                cumul_COV[j][i] = results_COV[j - 1][1]

    mean_AGA = std_AGA = mean_GA = std_GA = mean_RAN = std_RAN = mean_COV = std_COV = np.zeros(budget + 1)
    if algorithms == BASELINE_ALL or algorithms == BASELINE_AGA:
        mean_AGA = np.mean(cumul_AGA, axis=1)
        std_AGA = np.std(cumul_AGA, axis=1)

    if algorithms == BASELINE_ALL or algorithms == BASELINE_GA:
        mean_GA = np.mean(cumul_GA, axis=1)
        std_GA = np.std(cumul_GA, axis=1)

    if algorithms == BASELINE_ALL or algorithms == BASELINE_RAN:
        mean_RAN = np.mean(cumul_RAN, axis=1)
        std_RAN = np.std(cumul_RAN, axis=1)

    if algorithms == BASELINE_ALL or algorithms == BASELINE_COV:
        mean_COV = np.mean(cumul_COV, axis=1)
        std_COV = np.std(cumul_COV, axis=1)

    return np.array([mean_AGA, std_AGA, mean_GA,
                     std_GA, mean_RAN, std_RAN, mean_COV, std_COV])

def test_dartmouth_baseline(algorithms):
    '''The baseline (GA, random, coverage), without background, homogeneous, 40 x 40 grid
    '''
    cov_file            = 'dartmouth/cov'
    sensor_file         = 'dartmouth/sensors'
    intruder_hypo_file  = 'dartmouth/hypothesis'
    #selectsensor = SelectSensor(40)
    #selectsensor.init_data(cov_file, sensor_file, intruder_hypo_file)
    #selectsensor.rescale_intruder_hypothesis()
    #selectsensor.transmitters_to_array()        # for GPU
    budget = 20

    selectsensor = SelectSensor(64)
    cov_file_cur = cov_file
    sensor_file_cur = sensor_file
    intruder_hypo_file_cur = intruder_hypo_file
    #cov_file_cur = cov_file
    #sensor_file_cur = sensor_file
    #intruder_hypo_file_cur = intruder_hypo_file
    selectsensor.init_data(cov_file_cur, sensor_file_cur, intruder_hypo_file_cur)
    selectsensor.rescale_wifi_hypothesis()
    # print(selectsensor.means)
    selectsensor.transmitters_to_array()        # for GPU
    if algorithms == BASELINE_ALL or algorithms == BASELINE_AGA:
        #selectsensor.transmitters_to_array()
        results_AGA = selectsensor.select_offline_greedy_lazy_gpu(budget, 20, o_t_approx_kernal2)
        #print(results_AGA)
        #plots.save_data(results_AGA, 'plot_data_splat/fig1-homo/GA')

    if algorithms == BASELINE_ALL or algorithms == BASELINE_GA:  # GA
        #selectsensor.transmitters_to_array()
        results_GA = selectsensor.select_offline_GA(budget, o_t_iter_kernal)
        #plots.save_data(results_GA, 'plot_data_splat/fig1-homo/GA')

    if algorithms == BASELINE_ALL or algorithms == BASELINE_RAN:  # Random
        results_RAN = selectsensor.select_offline_random(budget, 20)
        #plots.save_data(results_RAN, 'plot_data_splat/fig1-homo/random')

    if algorithms == BASELINE_ALL or algorithms == BASELINE_COV:  # Coverage
        results_COV = selectsensor.select_offline_coverage(budget, 20)
        plots.save_data(results_COV, 'plot_data_splat/fig1-homo/coverage')

    for j in range(len(results_AGA)):
        if algorithms == BASELINE_ALL or algorithms == BASELINE_AGA:
            print(results_AGA[j][2], end=' ')
        if algorithms == BASELINE_ALL or algorithms == BASELINE_GA:
            print(results_GA[j][1], end=' ')
        if algorithms == BASELINE_ALL or algorithms == BASELINE_RAN:
            #print(cumul_RAN, results_RAN[i][1])
            print(results_RAN[j][1], end=' ')
        if algorithms == BASELINE_ALL or algorithms == BASELINE_COV:
            print(results_COV[j][1])
#print(cumul_AGA)
# cumul_AGA /= num_iterations
# cumul_GA /= num_iterations
# cumul_RAN /= num_iterations
# cumul_COV /= num_iterations

def test_utah_baseline(algorithms, num_iterations=20):
    '''The baseline (GA, random, coverage), without background, homogeneous, 40 x 40 grid
    '''
    cov_file            = 'cov'
    sensor_file         = 'sensors'
    intruder_hypo_file  = 'hypothesis'
    #selectsensor = SelectSensor(40)
    #selectsensor.init_data(cov_file, sensor_file, intruder_hypo_file)
    #selectsensor.rescale_intruder_hypothesis()
    #selectsensor.transmitters_to_array()        # for GPU
    budget = 10

    cumul_AGA = np.zeros((budget + 1, num_iterations))
    cumul_GA = np.zeros((budget + 1, num_iterations))
    cumul_RAN = np.zeros((budget + 1, num_iterations))
    cumul_COV = np.zeros((budget + 1, num_iterations))
    for i in range(num_iterations):
        selectsensor = SelectSensor(14)
        cov_file_cur = cov_file
        sensor_file_cur = sensor_file
        intruder_hypo_file_cur = intruder_hypo_file
        #cov_file_cur = cov_file
        #sensor_file_cur = sensor_file
        #intruder_hypo_file_cur = intruder_hypo_file
        selectsensor.init_data(cov_file_cur, sensor_file_cur, intruder_hypo_file_cur)
        selectsensor.rescale_intruder_hypothesis(noise_floor=WIFI_NOISE_FLOOR)
        # print(selectsensor.means)
        selectsensor.transmitters_to_array()        # for GPU
        if algorithms == BASELINE_ALL or algorithms == BASELINE_AGA:
            #selectsensor.transmitters_to_array()
            results_AGA = selectsensor.select_offline_greedy_lazy_gpu(budget, 20, o_t_approx_dist_kernal2)
            #print(results_AGA)
            #plots.save_data(results_AGA, 'plot_data_splat/fig1-homo/GA')

        if algorithms == BASELINE_ALL or algorithms == BASELINE_GA:  # GA
            #selectsensor.transmitters_to_array()
            results_GA = selectsensor.select_offline_GA(budget, o_t_iter_kernal)
            #plots.save_data(results_GA, 'plot_data_splat/fig1-homo/GA')

        if algorithms == BASELINE_ALL or algorithms == BASELINE_RAN:  # Random
            results_RAN = selectsensor.select_offline_random(budget, 20)
            #plots.save_data(results_RAN, 'plot_data_splat/fig1-homo/random')

        if algorithms == BASELINE_ALL or algorithms == BASELINE_COV:  # Coverage
            results_COV = selectsensor.select_offline_coverage(budget, 20)
            plots.save_data(results_COV, 'plot_data_splat/fig1-homo/coverage')

        for j in range(len(results_AGA)):
            print(j + 1, end = ' ')
            if algorithms == BASELINE_ALL or algorithms == BASELINE_AGA:
                cumul_AGA[j] = results_AGA[j-1][2] * 4.454545
            if algorithms == BASELINE_ALL or algorithms == BASELINE_GA:
                cumul_GA[j] = results_GA[j-1][1] * 4.454545
            if algorithms == BASELINE_ALL or algorithms == BASELINE_RAN:
                #print(cumul_RAN, results_RAN[i][1])
                cumul_RAN[j] = results_RAN[j-1][1] * 4.454545
                print('Random result', j, cumul_RAN[j-1])
                #print(results_RAN[j][1] * 4.454545, end=' ')
            if algorithms == BASELINE_ALL or algorithms == BASELINE_COV:
                cumul_COV[j] = results_COV[j-1][1] * 4.454545
    print('')
    for j in range(budget):
        print(j, end=' ')
        if algorithms == BASELINE_ALL or algorithms == BASELINE_AGA:
            mean_AGA = np.mean(cumul_AGA[j][:])
            std_AGA = np.std(cumul_AGA[j][:])
            print(mean_AGA, std_AGA, end=' ')
        if algorithms == BASELINE_ALL or algorithms == BASELINE_GA:
            mean_GA = np.mean(cumul_GA[j][:])
            std_GA = np.std(cumul_GA[j][:])
            print(mean_GA, std_GA, end=' ')
        if algorithms == BASELINE_ALL or algorithms == BASELINE_RAN:
            mean_RAN = np.mean(cumul_RAN[j][:])
            std_RAN = np.std(cumul_RAN[j][:])
            print(mean_RAN, std_RAN, end=' ')
        if algorithms == BASELINE_ALL or algorithms == BASELINE_COV:
            mean_COV = np.mean(cumul_COV[j][:])
            std_COV = np.std(cumul_COV[j][:])
            print(mean_COV, std_COV, end=' ')
        print('')

def test_weighted_utah_baseline(algorithms, num_iterations=10):
    grid_len = 14
    selectsensor = SelectSensor(grid_len)
    cov_file_cur = 'utah/cov'
    sensor_file_cur = 'utah/sensors'
    intruder_hypo_file_cur = 'utah/hypothesis'
    #cov_file_cur = cov_file
    #sensor_file_cur = sensor_file
    #intruder_hypo_file_cur = intruder_hypo_file
    selectsensor.init_data(cov_file_cur, sensor_file_cur, intruder_hypo_file_cur)
    selectsensor.rescale_intruder_hypothesis(noise_floor=-60)
    # print(selectsensor.means)
    selectsensor.transmitters_to_array()        # for GPU

    budget = 10
    cumul_WAGA = np.zeros((budget + 1, num_iterations))
    cumul_AGA = np.zeros((budget + 1, num_iterations))
    cumul_GA = np.zeros((budget + 1, num_iterations))
    cumul_COV = np.zeros((budget + 1, num_iterations))
    cumul_RAN = np.zeros((budget + 1, num_iterations))

    for i in range(num_iterations):
        if algorithms == BASELINE_ALL or algorithms == BASELINE_WAGA:
            results_WAGA = selectsensor.select_offline_greedy_lazy_gpu(budget, 20, o_t_approx_dist_kernal2)

        if algorithms == BASELINE_ALL or algorithms == BASELINE_AGA:
            results_AGA = selectsensor.select_offline_greedy_lazy_gpu(budget, 20, o_t_approx_kernal2)

        if algorithms == BASELINE_ALL or algorithms == BASELINE_GA:
            results_GA = selectsensor.select_offline_GA(budget, o_t_iter_kernal)
        # plots.save_data(results_GA, 'plot_data_splat/fig1-homo/GA')

        if algorithms == BASELINE_ALL or algorithms == BASELINE_RAN:  # Random
            results_RAN = selectsensor.select_offline_random(budget, 20)
            # plots.save_data(results_RAN, 'plot_data_splat/fig1-homo/random')

        if algorithms == BASELINE_ALL or algorithms == BASELINE_COV:  # Coverage
            results_COV = selectsensor.select_offline_coverage(budget, 20)

        true_tx = np.where(selectsensor.present == 1)[0]
        true_x = [i // grid_len for i in true_tx]
        true_y = [i % grid_len for i in true_tx]
        #print(true_tx, true_x, true_y)

        for j in range(1, budget+1):
            error_AGA = np.zeros(budget + 1)
            error_WAGA = np.zeros(budget + 1)
            error_GA = np.zeros(budget + 1)
            error_COV = np.zeros(budget + 1)
            error_RAN = np.zeros(budget + 1)

            if algorithms == BASELINE_ALL or algorithms == BASELINE_WAGA:
                error = 0
                for tno, trans in enumerate(true_x):
                    #print(results_WAGA[j][3], selectsensor.means_rescale.shape, trans, true_y[tno])

                    selectsensor.compute_conditional_error(trans, true_y[tno], results_WAGA[j-1][3])
                    #np.set_printoptions(np.infty)
                    #print(selectsensor.grid_posterior)
                    for x in range(selectsensor.grid_len):
                        for y in range(selectsensor.grid_len):
                            distance = np.sqrt((x - trans) ** 2 + (y - true_y[tno]) ** 2)
                            distance += 0.5
                            error += selectsensor.grid_posterior[grid_len * x + y] * distance

                    #error_WAGA[j] = selectsensor.compute_conditional_error(trans, true_y[tno],
                error_WAGA[j] = error

            if algorithms == BASELINE_ALL or algorithms == BASELINE_AGA:
                error = 0
                for tno, trans in enumerate(true_x):
                    #print(results_WAGA[j][3], selectsensor.means_rescale.shape, trans, true_y[tno])

                    selectsensor.compute_conditional_error(trans, true_y[tno], results_AGA[j-1][3])

                    for x in range(selectsensor.grid_len):
                        for y in range(selectsensor.grid_len):
                            distance = np.sqrt((x - trans) ** 2 + (y - true_y[tno]) ** 2)
                            distance += 0.5
                            error += selectsensor.grid_posterior[grid_len * x + y] * distance

                error_AGA[j] = error
            if algorithms == BASELINE_ALL or algorithms == BASELINE_GA:
                error = 0
                for tno, trans in enumerate(true_x):
                    #print(results_WAGA[j][3], selectsensor.means_rescale.shape, trans, true_y[tno])

                    selectsensor.compute_conditional_error(trans, true_y[tno], results_GA[j-1][2])

                    for x in range(selectsensor.grid_len):
                        for y in range(selectsensor.grid_len):
                            distance = np.sqrt((x - trans) ** 2 + (y - true_y[tno]) ** 2)
                            distance += 0.5
                            error += selectsensor.grid_posterior[grid_len * x + y] * distance
                error_GA[j] = error
            if algorithms == BASELINE_ALL or algorithms == BASELINE_COV:
                error = 0
                for tno, trans in enumerate(true_x):
                    # print(results_WAGA[j][3], selectsensor.means_rescale.shape, trans, true_y[tno])

                    selectsensor.compute_conditional_error(trans, true_y[tno], results_COV[j - 1][2])

                    for x in range(selectsensor.grid_len):
                        for y in range(selectsensor.grid_len):
                            distance = np.sqrt((x - trans) ** 2 + (y - true_y[tno]) ** 2)
                            distance += 0.5
                            error += selectsensor.grid_posterior[grid_len * x + y] * distance
                error_COV[j] = error

            if algorithms == BASELINE_ALL or algorithms == BASELINE_RAN:
                error = 0
                for tno, trans in enumerate(true_x):
                    # print(results_WAGA[j][3], selectsensor.means_rescale.shape, trans, true_y[tno])

                    selectsensor.compute_conditional_error(trans, true_y[tno], results_RAN[j - 1][2])

                    for x in range(selectsensor.grid_len):
                        for y in range(selectsensor.grid_len):
                            distance = np.sqrt((x - trans) ** 2 + (y - true_y[tno]) ** 2)
                            distance += 0.5
                            error += selectsensor.grid_posterior[grid_len * x + y] * distance
                error_RAN[j] = error

            error_AGA /= len(true_tx)
            error_WAGA /= len(true_tx)
            error_GA /= len(true_tx)
            error_COV /= len(true_tx)
            error_RAN /= len(true_tx)
            #print(j, error_AGA[j], error_WAGA[j], error_GA[j], error_COV[j], error_RAN[j])
            print(j, end = ' ')
            if algorithms == BASELINE_ALL or algorithms == BASELINE_WAGA:
                cumul_WAGA[j][i] = error_WAGA[j]
            if algorithms == BASELINE_ALL or algorithms == BASELINE_AGA:
                cumul_AGA[j][i] = error_AGA[j]
            if algorithms == BASELINE_ALL or algorithms == BASELINE_GA:
                cumul_GA[j][i] = error_GA[j]
            if algorithms == BASELINE_ALL or algorithms == BASELINE_RAN:
                #print(cumul_RAN, results_RAN[i][1])
                cumul_RAN[j][i] = error_RAN[j]
            if algorithms == BASELINE_ALL or algorithms == BASELINE_COV:
                cumul_COV[j][i] = error_COV[j]
    for j in range(budget):
        print(j, end=' ')
        if algorithms == BASELINE_ALL or algorithms == BASELINE_AGA:
            mean_AGA = np.mean(cumul_WAGA[j][:])
            std_AGA = np.std(cumul_WAGA[j][:])
            print(mean_AGA, std_AGA, end=' ')
        if algorithms == BASELINE_ALL or algorithms == BASELINE_GA:
            mean_GA = np.mean(cumul_GA[j][:])
            std_GA = np.std(cumul_GA[j][:])
            print(mean_GA, std_GA, end=' ')
        if algorithms == BASELINE_ALL or algorithms == BASELINE_RAN:
            mean_RAN = np.mean(cumul_RAN[j][:])
            std_RAN = np.std(cumul_RAN[j][:])
            print(mean_RAN, std_RAN, end=' ')
        if algorithms == BASELINE_ALL or algorithms == BASELINE_COV:
            mean_COV = np.mean(cumul_COV[j][:])
            std_COV = np.std(cumul_COV[j][:])
            print(mean_COV, std_COV, end=' ')
        print('')

def test_splat_hetero_baseline(large, algorithms, num_iterations=1):
    if large is STD_INSTANCE:
        cov_file            = 'dataSplat/1600-100/cov'
        sensor_file         = 'dataSplat/1600-100/sensors'
        intruder_hypo_file  = 'dataSplat/1600-100/hypothesis'
        #selectsensor = SelectSensor(40)
        #selectsensor.init_data(cov_file, sensor_file, intruder_hypo_file)
        #selectsensor.rescale_intruder_hypothesis()
        #selectsensor.transmitters_to_array()        # for GPU
        budget = 20
    else:
        config = 'config/splat_config_64.json'
        cov_file = 'dataSplat/4096/cov'
        sensor_file = 'dataSplat/4096/sensors'
        intruder_hypo_file  = 'dataSplat/4096/hypothesis'
        selectsensor = SelectSensor(64)
        selectsensor.init_data(cov_file, sensor_file, intruder_hypo_file)
        selectsensor.rescale_intruder_hypothesis()
        #selectsensor.transmitters_to_array()  # for GPU
        budget = 30

    cumul_AGA = np.zeros((budget, num_iterations))
    cumul_GA = np.zeros((budget, num_iterations))
    cumul_RAN = np.zeros((budget, num_iterations))
    cumul_COV = np.zeros((budget, num_iterations))
    for i in range(num_iterations):
        gen_data(i, num_sensors=100, grid_len=40, hetero=True)
        selectsensor = SelectSensor(40)
        cov_file_cur = cov_file + str(i)
        sensor_file_cur = sensor_file + str(i)
        intruder_hypo_file_cur = intruder_hypo_file + str(i)
        #cov_file_cur = cov_file
        #sensor_file_cur = sensor_file
        #intruder_hypo_file_cur = intruder_hypo_file
        selectsensor.init_data(cov_file_cur, sensor_file_cur, intruder_hypo_file_cur)

        selectsensor.rescale_intruder_hypothesis()
        # print(selectsensor.means)
        selectsensor.transmitters_to_array()        # for GPU
        budget = 10
        if algorithms == BASELINE_ALL or algorithms == BASELINE_AGA:
            #selectsensor.transmitters_to_array()
            results_AGA = selectsensor.select_offline_greedy_hetero(budget, 20, o_t_approx_kernal2)
            costs_AGA = [result[0] for result in results_AGA]
            #print(results_AGA)
            #plots.save_data(results_AGA, 'plot_data_splat/fig1-homo/GA')

        if algorithms == BASELINE_ALL or algorithms == BASELINE_GA:  # GA
            #selectsensor.transmitters_to_array()
            results_GA = selectsensor.select_offline_GA_hetero(budget, o_t_iter_kernal)
            costs_GA = [result[0] for result in results_GA]
            #plots.save_data(results_GA, 'plot_data_splat/fig1-homo/GA')

        if algorithms == BASELINE_ALL or algorithms == BASELINE_RAN:  # Random
            results_RAN = selectsensor.select_offline_random_hetero(budget, 20)
            costs_RAN = [result[0] for result in results_RAN]
            #plots.save_data(results_RAN, 'plot_data_splat/fig1-homo/random')

        if algorithms == BASELINE_ALL or algorithms == BASELINE_COV:  # Coverage
            results_COV = selectsensor.select_offline_coverage_hetero(budget, 20)
            costs_COV = [result[0] for result in results_RAN]
            #plots.save_data(results_COV, 'plot_data_splat/fig1-homo/coverage')

        AGA_ptr = 0
        GA_ptr = 0
        RAN_ptr = 0
        COV_ptr = 0
        print(costs_AGA)
        for j in range(1, budget + 1):
            if algorithms == BASELINE_ALL or algorithms == BASELINE_AGA:
                try:
                    while costs_AGA[AGA_ptr] < j:
                        AGA_ptr += 1
                    cumul_AGA[j][i] = results_AGA[AGA_ptr - 1][1]
                except:
                    cumul_AGA[j][i] = results_AGA[-1][1]

            if algorithms == BASELINE_ALL or algorithms == BASELINE_GA:
                try:
                    while costs_GA[GA_ptr] < j:
                        GA_ptr += 1
                    cumul_GA[j][i] = results_GA[GA_ptr - 1][1]
                except:
                    cumul_GA[j][i] = results_GA[-1][1]
            if algorithms == BASELINE_ALL or algorithms == BASELINE_RAN:
                #print(cumul_RAN, results_RAN[i][1])
                try:
                    while costs_RAN[RAN_ptr] < j:
                        RAN_ptr += 1
                    cumul_RAN[j][i] = results_RAN[RAN_ptr - 1][1]
                except:
                    cumul_RAN[j][i] = results_RAN[-1][1]
                cumul_RAN[j][i] = results_RAN[j][1]
            if algorithms == BASELINE_ALL or algorithms == BASELINE_COV:
                try:
                    while costs_COV[COV_ptr] < j:
                        COV_ptr += 1
                    cumul_COV[j][i] = results_COV[COV_ptr - 1][1]
                except:
                    cumul_COV[j][i] = results_COV[-1][1]
                cumul_COV[j][i] = results_COV[j][1]
    #print(cumul_AGA)
    # cumul_AGA /= num_iterations
    # cumul_GA /= num_iterations
    # cumul_RAN /= num_iterations
    # cumul_COV /= num_iterations

    print('')
    for j in range(budget + 1):
        print(j, end = ' ')
        if algorithms == BASELINE_ALL or algorithms == BASELINE_AGA:
            mean_AGA = np.mean(cumul_AGA[j][:])
            std_AGA = np.std(cumul_AGA[j][:])
            print(mean_AGA, std_AGA, end = ' ')
        if algorithms == BASELINE_ALL or algorithms == BASELINE_GA:
            mean_GA = np.mean(cumul_GA[j][:])
            std_GA = np.std(cumul_GA[j][:])
            print(mean_GA, std_GA, end=' ')
        if algorithms == BASELINE_ALL or algorithms == BASELINE_RAN:
            mean_RAN = np.mean(cumul_RAN[j][:])
            std_RAN = np.std(cumul_RAN[j][:])
            print(mean_RAN, std_RAN, end=' ')
        if algorithms == BASELINE_ALL or algorithms == BASELINE_COV:
            mean_COV = np.mean(cumul_COV[j][:])
            std_COV = np.std(cumul_COV[j][:])
            print(mean_COV, std_COV, end=' ')
        print('')

def test_weighted_baseline(large, algorithms, num_iterations=1):
    if large is SMALL_INSTANCE:
        cov_file            = 'dataSplat/100/cov'
        sensor_file         = 'dataSplat/100/sensors'
        intruder_hypo_file  = 'dataSplat/100/hypothesis'
        #selectsensor = SelectSensor(40)
        #selectsensor.init_data(cov_file, sensor_file, intruder_hypo_file)
        #selectsensor.rescale_intruder_hypothesis()
        #selectsensor.transmitters_to_array()        # for GPU
        budget = 10
        grid_len = 10
    if large is STD_INSTANCE:
        cov_file            = 'dataSplat/1600-100/cov'
        sensor_file         = 'dataSplat/1600-100/sensors'
        intruder_hypo_file  = 'dataSplat/1600-100/hypothesis'
        #selectsensor = SelectSensor(40)
        #selectsensor.init_data(cov_file, sensor_file, intruder_hypo_file)
        #selectsensor.rescale_intruder_hypothesis()
        #selectsensor.transmitters_to_array()        # for GPU
        budget = 20
        grid_len = 40
    elif large is LARGE_INSTANCE:
        config = 'config/splat_config_64.json'
        cov_file = 'dataSplat/4096/cov'
        sensor_file = 'dataSplat/4096/sensors'
        intruder_hypo_file  = 'dataSplat/4096/hypothesis'
        selectsensor = SelectSensor(64)
        selectsensor.init_data(cov_file, sensor_file, intruder_hypo_file)
        selectsensor.rescale_intruder_hypothesis()
        grid_len = 64
        #selectsensor.transmitters_to_array()  # for GPU
        budget = 30


    cumul_WAGA = np.zeros((budget + 1, num_iterations))
    cumul_AGA = np.zeros((budget + 1, num_iterations))
    cumul_GA = np.zeros((budget + 1, num_iterations))
    cumul_RAN = np.zeros((budget + 1, num_iterations))
    cumul_COV = np.zeros((budget + 1, num_iterations))
    for i in range(num_iterations):
        gen_data(i)
        selectsensor = SelectSensor(grid_len)
        cov_file_cur = cov_file + str(i)
        sensor_file_cur = sensor_file + str(i)
        intruder_hypo_file_cur = intruder_hypo_file + str(i)
        #cov_file_cur = cov_file
        #sensor_file_cur = sensor_file
        #intruder_hypo_file_cur = intruder_hypo_file
        selectsensor.init_data(cov_file_cur, sensor_file_cur, intruder_hypo_file_cur)
        selectsensor.rescale_intruder_hypothesis()
        # print(selectsensor.means)
        selectsensor.transmitters_to_array()        # for GPU

        if algorithms == BASELINE_ALL or algorithms == BASELINE_WAGA:
            results_WAGA = selectsensor.select_offline_greedy_lazy_gpu(budget, 20, o_t_approx_dist_kernal2)

        if algorithms == BASELINE_ALL or algorithms == BASELINE_AGA:
            results_AGA = selectsensor.select_offline_greedy_lazy_gpu(budget, 20, o_t_approx_kernal2)

        if algorithms == BASELINE_ALL or algorithms == BASELINE_GA:
            results_GA = selectsensor.select_offline_GA(budget, o_t_iter_kernal)
        # plots.save_data(results_GA, 'plot_data_splat/fig1-homo/GA')

        if algorithms == BASELINE_ALL or algorithms == BASELINE_RAN:  # Random
            results_RAN = selectsensor.select_offline_random(budget, 20)
            # plots.save_data(results_RAN, 'plot_data_splat/fig1-homo/random')

        if algorithms == BASELINE_ALL or algorithms == BASELINE_COV:  # Coverage
            results_COV = selectsensor.select_offline_coverage(budget, 20)

        true_x = np.random.choice(range(grid_len), size=100, replace=True)
        true_y = np.random.choice(range(grid_len), size=100, replace=True)
        for j in range(1, budget+1):
            error_AGA = np.zeros(budget + 1)
            error_WAGA = np.zeros(budget + 1)
            error_GA = np.zeros(budget + 1)
            error_COV = np.zeros(budget + 1)
            error_RAN = np.zeros(budget + 1)

            for tno, trans in enumerate(true_x):
                if algorithms == BASELINE_ALL or algorithms == BASELINE_WAGA:
                    print(results_WAGA[j - 1][3], selectsensor.means_rescale.shape, trans, true_y[tno])
                    error_WAGA[j] += selectsensor.compute_conditional_error(trans, true_y[tno],
                                                                        results_WAGA[j-1][3])
                if algorithms == BASELINE_ALL or algorithms == BASELINE_AGA:
                    error_AGA[j] += selectsensor.compute_conditional_error(trans, true_y[tno],
                                                                    results_AGA[j-1][3])
                if algorithms == BASELINE_ALL or algorithms == BASELINE_GA:
                    error_GA[j] += selectsensor.compute_conditional_error(trans, true_y[tno],
                                                                       results_GA[j-1][2])
                if algorithms == BASELINE_ALL or algorithms == BASELINE_COV:
                    error_COV[j] += selectsensor.compute_conditional_error(trans, true_y[tno],
                                                                           results_COV[j-1][2])
                if algorithms == BASELINE_ALL or algorithms == BASELINE_RAN:
                    error_RAN[j] += selectsensor.compute_conditional_error(trans, true_y[tno],
                                                                       results_RAN[j-1][2])
            error_AGA /= 100
            error_WAGA /= 100
            error_GA /= 100
            error_COV /= 100
            error_RAN /= 100
            #print(j, error_AGA[j], error_WAGA[j], error_GA[j], error_COV[j], error_RAN[j])

            if algorithms == BASELINE_ALL or algorithms == BASELINE_WAGA:
                cumul_WAGA[j][i] = error_WAGA[j]
            if algorithms == BASELINE_ALL or algorithms == BASELINE_AGA:
                cumul_AGA[j][i] = error_AGA[j]
            if algorithms == BASELINE_ALL or algorithms == BASELINE_GA:
                cumul_GA[j][i] = error_GA[j]
            if algorithms == BASELINE_ALL or algorithms == BASELINE_RAN:
                #print(cumul_RAN, results_RAN[i][1])
                cumul_RAN[j][i] = error_RAN[j]
            if algorithms == BASELINE_ALL or algorithms == BASELINE_COV:
                cumul_COV[j][i] = error_COV[j]

    print('')
    for j in range(budget + 1):
        print(j, end = ' ')
        if algorithms == BASELINE_ALL or algorithms == BASELINE_WAGA:
            mean_WAGA = np.mean(cumul_WAGA[j][:])
            std_WAGA = np.std(cumul_WAGA[j][:])
            print(mean_WAGA, std_WAGA, end = ' ')
        if algorithms == BASELINE_ALL or algorithms == BASELINE_AGA:
            mean_AGA = np.mean(cumul_AGA[j][:])
            std_AGA = np.std(cumul_AGA[j][:])
            print(mean_AGA, std_AGA, end = ' ')
        if algorithms == BASELINE_ALL or algorithms == BASELINE_GA:
            mean_GA = np.mean(cumul_GA[j][:])
            std_GA = np.std(cumul_GA[j][:])
            print(mean_GA, std_GA, end=' ')
        if algorithms == BASELINE_ALL or algorithms == BASELINE_RAN:
            mean_RAN = np.mean(cumul_RAN[j][:])
            std_RAN = np.std(cumul_RAN[j][:])
            print(mean_RAN, std_RAN, end=' ')
        if algorithms == BASELINE_ALL or algorithms == BASELINE_COV:
            mean_COV = np.mean(cumul_COV[j][:])
            std_COV = np.std(cumul_COV[j][:])
            print(mean_COV, std_COV, end=' ')
        print('')

def test_weighted_hetero_baseline(large, algorithms, num_iterations=1):
    if large is False:
        cov_file            = 'dataSplat/1600-100/cov'
        sensor_file         = 'dataSplat/1600-100/sensors'
        intruder_hypo_file  = 'dataSplat/1600-100/hypothesis'
        #selectsensor = SelectSensor(40)
        #selectsensor.init_data(cov_file, sensor_file, intruder_hypo_file)
        #selectsensor.rescale_intruder_hypothesis()
        #selectsensor.transmitters_to_array()        # for GPU
        budget = 15
        grid_len = 40
    else:
        config = 'config/splat_config_64.json'
        cov_file = 'dataSplat/4096/cov'
        sensor_file = 'dataSplat/4096/sensors'
        intruder_hypo_file  = 'dataSplat/4096/hypothesis'
        selectsensor = SelectSensor(64)
        selectsensor.init_data(cov_file, sensor_file, intruder_hypo_file)
        selectsensor.rescale_intruder_hypothesis()
        grid_len = 64
        #selectsensor.transmitters_to_array()  # for GPU
        budget = 30

    cumul_WAGA = np.zeros((budget + 1, num_iterations))
    cumul_AGA = np.zeros((budget + 1, num_iterations))
    cumul_GA = np.zeros((budget + 1, num_iterations))
    cumul_RAN = np.zeros((budget + 1, num_iterations))
    cumul_COV = np.zeros((budget + 1, num_iterations))
    for i in range(num_iterations):
        gen_data(i, hetero=True)
        selectsensor = SelectSensor(grid_len)
        cov_file_cur = cov_file + str(i)
        sensor_file_cur = sensor_file + str(i)
        intruder_hypo_file_cur = intruder_hypo_file + str(i)
        #cov_file_cur = cov_file
        #sensor_file_cur = sensor_file
        #intruder_hypo_file_cur = intruder_hypo_file
        selectsensor.init_data(cov_file_cur, sensor_file_cur, intruder_hypo_file_cur)
        selectsensor.rescale_intruder_hypothesis()
        # print(selectsensor.means)
        selectsensor.transmitters_to_array()        # for GPU
        if algorithms == BASELINE_ALL or algorithms == BASELINE_AGA:
            #selectsensor.transmitters_to_array()
            if algorithms == BASELINE_ALL or algorithms == BASELINE_WAGA:
                results_WAGA = selectsensor.select_offline_greedy_hetero(budget, 20, o_t_approx_dist_kernal2)

            if algorithms == BASELINE_ALL or algorithms == BASELINE_AGA:
                results_AGA = selectsensor.select_offline_greedy_hetero(budget, 20, o_t_approx_kernal2)

            if algorithms == BASELINE_ALL or algorithms == BASELINE_GA:
                results_GA = selectsensor.select_offline_GA_hetero(budget, o_t_iter_kernal)
            # plots.save_data(results_GA, 'plot_data_splat/fig1-homo/GA')

            if algorithms == BASELINE_ALL or algorithms == BASELINE_RAN:  # Random
                results_RAN = selectsensor.select_offline_random(budget, 20)
                # plots.save_data(results_RAN, 'plot_data_splat/fig1-homo/random')

            if algorithms == BASELINE_ALL or algorithms == BASELINE_COV:  # Coverage
                results_COV = selectsensor.select_offline_coverage_hetero(budget, 20)

            true_x = np.random.choice(range(grid_len), size=100, replace=True)
            true_y = np.random.choice(range(grid_len), size=100, replace=True)
            for j in range(1, budget+1):
                error_AGA = np.zeros(budget + 1)
                error_WAGA = np.zeros(budget + 1)
                error_GA = np.zeros(budget + 1)
                error_COV = np.zeros(budget + 1)
                error_RAN = np.zeros(budget + 1)

                for tno, trans in enumerate(true_x):
                    if algorithms == BASELINE_ALL or algorithms == BASELINE_WAGA:
                        error_WAGA[j] += selectsensor.compute_conditional_error(trans, true_y[tno],
                                                                            results_WAGA[j-1][2])
                    if algorithms == BASELINE_ALL or algorithms == BASELINE_AGA:
                        error_AGA[j] += selectsensor.compute_conditional_error(trans, true_y[tno],
                                                                        results_AGA[j-1][2])
                    if algorithms == BASELINE_ALL or algorithms == BASELINE_GA:
                        error_GA[j] += selectsensor.compute_conditional_error(trans, true_y[tno],
                                                                           results_GA[j-1][2])
                    if algorithms == BASELINE_ALL or algorithms == BASELINE_COV:
                        error_COV[j] += selectsensor.compute_conditional_error(trans, true_y[tno],
                                                                               results_COV[j-1][2])
                    if algorithms == BASELINE_ALL or algorithms == BASELINE_RAN:
                        error_RAN[j] += selectsensor.compute_conditional_error(trans, true_y[tno],
                                                                           results_RAN[j-1][2])
                error_AGA /= 100
                error_WAGA /= 100
                error_GA /= 100
                error_COV /= 100
                error_RAN /= 100
                #print(j, error_AGA[j], error_WAGA[j], error_GA[j], error_COV[j], error_RAN[j])

                if algorithms == BASELINE_ALL or algorithms == BASELINE_WAGA:
                    cumul_WAGA[j][i] = error_WAGA[j]
                if algorithms == BASELINE_ALL or algorithms == BASELINE_AGA:
                    cumul_AGA[j][i] = error_AGA[j]
                if algorithms == BASELINE_ALL or algorithms == BASELINE_GA:
                    cumul_GA[j][i] = error_GA[j]
                if algorithms == BASELINE_ALL or algorithms == BASELINE_RAN:
                    #print(cumul_RAN, results_RAN[i][1])
                    cumul_RAN[j][i] = error_RAN[j]
                if algorithms == BASELINE_ALL or algorithms == BASELINE_COV:
                    cumul_COV[j][i] = error_COV[j]

    print('')
    for j in range(budget + 1):
        print(j, end = ' ')
        if algorithms == BASELINE_ALL or algorithms == BASELINE_WAGA:
            mean_WAGA = np.mean(cumul_WAGA[j][:])
            std_WAGA = np.std(cumul_WAGA[j][:])
            print(mean_WAGA, std_WAGA, end = ' ')
        if algorithms == BASELINE_ALL or algorithms == BASELINE_AGA:
            mean_AGA = np.mean(cumul_AGA[j][:])
            std_AGA = np.std(cumul_AGA[j][:])
            print(mean_AGA, std_AGA, end = ' ')
        if algorithms == BASELINE_ALL or algorithms == BASELINE_GA:
            mean_GA = np.mean(cumul_GA[j][:])
            std_GA = np.std(cumul_GA[j][:])
            print(mean_GA, std_GA, end=' ')
        if algorithms == BASELINE_ALL or algorithms == BASELINE_RAN:
            mean_RAN = np.mean(cumul_RAN[j][:])
            std_RAN = np.std(cumul_RAN[j][:])
            print(mean_RAN, std_RAN, end=' ')
        if algorithms == BASELINE_ALL or algorithms == BASELINE_COV:
            mean_COV = np.mean(cumul_COV[j][:])
            std_COV = np.std(cumul_COV[j][:])
            print(mean_COV, std_COV, end=' ')
        print('')

def test_splat_opt():
    '''Comparing AGA to the optimal and baselines, without background, homogeneous, small grid 10 x 10
    '''
    config              = 'config/splat_config_10.json'
    # cov_file            = 'dataSplat/100/cov'
    # sensor_file         = 'dataSplat/100/sensors'
    # intruder_hypo_file  = 'dataSplat/100/hypothesis'
    cov_file = 'dataSplat/1600-100/cov'
    sensor_file = 'dataSplat/1600-100/sensors'
    intruder_hypo_file = 'dataSplat/1600-100/hypothesis'
    budget = 8
    num_iterations = 100
    cumul_GA = np.zeros((budget + 1, num_iterations))
    cumul_OPT = np.zeros((budget + 1, num_iterations))
    for i in range(0, num_iterations):
        print('\ncase {}'.format(i))
        selectsensor = SelectSensor(40)
        cov_file_cur = cov_file + str(i)
        sensor_file_cur = sensor_file + str(i)
        hypo_file = intruder_hypo_file + str(i)
        selectsensor.init_data(cov_file_cur, sensor_file_cur, hypo_file)
        selectsensor.rescale_intruder_hypothesis()
        selectsensor.transmitters_to_array()        # for GPU

        #results_AGA = selectsensor.select_offline_greedy_lazy_gpu(budget, 12, o_t_approx_kernal2)
        #print(results)
        # plots.save_data_AGA(results, 'plot_data_splat/fig2-homo-small/AGA{}'.format(i))

        results_GA = selectsensor.select_offline_GA(budget, o_t_iter_kernal)

        # results = selectsensor.select_offline_GA(10, 10, o_t_iter_kernal)
        
        #results = selectsensor.select_offline_coverage(10, 10)
        #plots.save_data(results, 'plot_data_splat/fig2-homo-small/coverage{}'.format(i))

        #results = selectsensor.select_offline_random(10, 10)
        #plots.save_data(results, 'plot_data_splat/fig2-homo-small/random{}'.format(i))

        # plot_data = []
        ots = [0] * (budget + 1)
        for budget in range(8, budget + 1):
            _, ots[budget] = selectsensor.select_offline_optimal(budget, 20)
        for j in range(1, budget + 1):
            cumul_GA[j][i] = results_GA[j-1][1]
            cumul_OPT[j][i] = ots[j]
    #difference = np.zeros(budget + 1)
    #difference = cumul_OPT[j, :]
    for j in range(1, budget + 1):
        mean_GA = np.mean(cumul_GA[j, :])
        mean_OPT = np.mean(cumul_OPT[j, :])

        print(j, mean_GA, mean_OPT)

def test_splat_localization_single_intruder():
    config              = 'config/splat_config_40.json'
    cov_file            = 'dataSplat/homogeneous-100/cov'
    sensor_file         = 'dataSplat/homogeneous-100/sensors'
    intruder_hypo_file  = 'dataSplat/homogeneous-100/hypothesis'
    primary_hypo_file   = 'dataSplat/homogeneous-100/hypothesis_primary'
    intr_pri_hypo_file  = 'dataSplat/homogeneous-100/hypothesis_intru_pri'
    secondary_hypo_file = 'dataSplat/homogeneous-100/hypothesis_secondary'
    all_hypo_file       = 'dataSplat/homogeneous-100/hypothesis_all'

    selectsensor = SelectSensor(40)
    selectsensor.init_data(cov_file, sensor_file, intruder_hypo_file)
    selectsensor.rescale_intruder_hypothesis()
    selectsensor.transmitters_to_array()
    #results = selectsensor.localize(10, -1)
    for r in results:
        print(r)

def test_splat_scalability(large = False):
    if large is False:
        cov_file = 'dataSplat/100/cov0'
        sensor_file = 'dataSplat/100/sensors0'
        intruder_hypo_file = 'dataSplat/100/hypothesis0'
        selectsensor = SelectSensor(10)
        budget = 10
    else:
        config = 'config/splat_config_64.json'
        cov_file = 'dataSplat/4096/cov'
        sensor_file = 'dataSplat/4096/sensors'
        intruder_hypo_file = 'dataSplat/4096/hypothesis'
        selectsensor = SelectSensor(64)
        selectsensor.init_data(cov_file, sensor_file, intruder_hypo_file)
        selectsensor.rescale_intruder_hypothesis()
        # selectsensor.transmitters_to_array()  # for GPU
        budget = 30

    gen_data(1, 100, subdir='dataSplat/100/', grid_len=10)
    selectsensor.init_data(cov_file, sensor_file, intruder_hypo_file)
    selectsensor.rescale_intruder_hypothesis()
    print('initial priori = ', selectsensor.grid_priori[0][0])
    # print(selectsensor.means)
    selectsensor.transmitters_to_array()  # for GPU
        # selectsensor.transmitters_to_array()
    AGA_small_start_opt = time.time()
    results_AGA = selectsensor.select_offline_greedy_lazy_gpu(budget, 20, o_t_approx_kernal2)
    print('final priori = ', selectsensor.grid_priori[0][0])
    AGA_small_end_opt = time.time()
    opt_time = AGA_small_end_opt - AGA_small_start_opt
    print('opt_time = ', opt_time)
    #
    results_AGA = selectsensor.select_offline_greedy_lazy_old(budget, 20, o_t_approx_kernal)
    AGA_small_end_no_opt = time.time()
    no_opt_time = AGA_small_end_opt - AGA_small_end_no_opt
    print('no_opt_time = ', no_opt_time)

    AGA_cpu_end = selectsensor.select_offline_greedy_p_lazy_cpu_old(budget, 20)
    cpu_end_time = time.time()
    cpu_time = cpu_end_time - no_opt_time

    # AGA_cpu_start = time.time()
    # results_cpu = selectsensor.select_offline_greedy_p_lazy_cpu(20, 20)
    # AGA_cpu_end = time.time()
    # cpu_time = AGA_cpu_end - AGA_cpu_start
    # print('CPU time = ', cpu_time)

from transmitter import Transmitter
def test_update_hypothesis():
    true_transmitter = Transmitter(5, 5)

    selectsensor = SelectSensor(10)
    cov_file = 'dataSplat/100/cov1'
    sensor_file = 'dataSplat/100/sensors1'
    intruder_hypo_file = 'dataSplat/100/hypothesis1'
    selectsensor.init_data(cov_file, sensor_file, intruder_hypo_file)
    selectsensor.rescale_intruder_hypothesis()
    selectsensor.transmitters_to_array()  # for GPU

    #results_AGA = selectsensor.select_offline_greedy_lazy_gpu(30, 20, o_t_approx_kernal2)
    #print('results_AGA = ', results_AGA[-1][3])
    selectsensor.update_hypothesis(true_transmitter, range(10))
    max_posterior = np.argmax(selectsensor.grid_posterior)
    print(selectsensor.grid_posterior[0][0], np.sum(selectsensor.grid_posterior))
    print(max_posterior)

def test_approx_ratio(size_instance, num_iterations = 1):
    if size_instance is LARGE_INSTANCE:
        subdir = 'dataSplat/4096/'
        # selectsensor = SelectSensor(40)
        # selectsensor.init_data(cov_file, sensor_file, intruder_hypo_file)
        # selectsensor.rescale_intruder_hypothesis()
        # selectsensor.transmitters_to_array()        # for GPU
        grid_len = 64
        budget = 40
    elif size_instance is STD_INSTANCE:
        subdir = 'dataSplat/1600-100/'
        # selectsensor.transmitters_to_array()  # for GPU
        grid_len = 40
        budget = 20
    elif size_instance is SMALL_INSTANCE:
        subdir = 'dataSplat/100/'
        grid_len = 10
        budget = 15

    cov_file = subdir + 'cov'
    sensor_file = subdir + 'sensors'
    intruder_hypo_file = subdir + 'hypothesis'

    ot_actual = np.zeros((num_iterations, budget + 1))
    ot_approx = np.zeros((num_iterations, budget + 1))
    for i in range(num_iterations):
        gen_data(i, 100, subdir=subdir, grid_len=grid_len)
        selectsensor = SelectSensor(grid_len)
        cov_file_cur = cov_file + str(i)
        sensor_file_cur = sensor_file + str(i)
        intruder_hypo_file_cur = intruder_hypo_file + str(i)
        # cov_file_cur = cov_file
        # sensor_file_cur = sensor_file
        # intruder_hypo_file_cur = intruder_hypo_file
        selectsensor.init_data(cov_file_cur, sensor_file_cur, intruder_hypo_file_cur)
        selectsensor.rescale_intruder_hypothesis()
        # print(selectsensor.means)
        selectsensor.transmitters_to_array()  # for GPU
        results_AGA = selectsensor.select_offline_greedy_lazy_gpu(budget, 20, o_t_approx_kernal2)
        for j in range(1, budget + 1):
            ot_actual[i, j] = results_AGA[j-1][2]
            ot_approx[i, j] = results_AGA[j-1][1]
    print(ot_actual, ot_approx)
    for j in range(1, budget + 1):
        print(j, end = ' ')
        print(np.mean(ot_actual[:, j]), end = ' ')
        print(np.std(ot_actual[:, j]), end = ' ')
        print(np.mean(ot_approx[:, j]), end = ' ')
        print(np.std(ot_approx[:, j]))

def test_outdoor_baseline(algorithms):
    grid_len = 10
    selectsensor = SelectSensor(grid_len)
    cov_file_cur = '10.6.testbed.inter-ildw-sub/cov'
    sensor_file_cur = '10.6.testbed.inter-ildw-sub/sensors'
    intruder_hypo_file_cur = '10.6.testbed.inter-ildw-sub/hypothesis'
    selectsensor.init_data(cov_file_cur, sensor_file_cur, intruder_hypo_file_cur)
    selectsensor.rescale_intruder_hypothesis(noise_floor=-48)
    selectsensor.transmitters_to_array()  # for GPU
    budget = 18
    if algorithms == BASELINE_ALL or algorithms == BASELINE_AGA:
        results_AGA = selectsensor.select_offline_greedy_lazy_gpu(budget, 10, o_t_approx_kernal2)
        # print(results_AGA)
        # plots.save_data(results_AGA, 'plot_data_splat/fig1-homo/GA')

    if algorithms == BASELINE_ALL or algorithms == BASELINE_GA:  # GA
        results_GA = selectsensor.select_offline_GA(budget, o_t_iter_kernal)
        # plots.save_data(results_GA, 'plot_data_splat/fig1-homo/GA')

    if algorithms == BASELINE_ALL or algorithms == BASELINE_RAN:  # Random
        results_RAN = selectsensor.select_offline_random(budget, 10)
        # plots.save_data(results_RAN, 'plot_data_splat/fig1-homo/random')

    if algorithms == BASELINE_ALL or algorithms == BASELINE_COV:  # Coverage
        results_COV = selectsensor.select_offline_coverage(budget, 10)

    for i in range(budget):
        print(i, results_AGA[i][2], results_GA[i][1], results_RAN[i][1], results_COV[i][1])


def test_outdoor_weighted(algorithms):
    grid_len = 10
    selectsensor = SelectSensor(grid_len)
    cov_file_cur = 'cov'
    sensor_file_cur = 'sensor'
    intruder_hypo_file_cur = 'hypothesis'
    budget = 15
    # cov_file_cur = cov_file
    # sensor_file_cur = sensor_file
    # intruder_hypo_file_cur = intruder_hypo_file
    selectsensor.init_data(cov_file_cur, sensor_file_cur, intruder_hypo_file_cur)
    selectsensor.rescale_intruder_hypothesis(noise_floor=-40)
    # print(selectsensor.means)
    selectsensor.transmitters_to_array()        # for GPU

    if algorithms == BASELINE_ALL or algorithms == BASELINE_WAGA:
        results_WAGA = selectsensor.select_offline_greedy_lazy_gpu(budget, 20, o_t_approx_dist_kernal2)

    if algorithms == BASELINE_ALL or algorithms == BASELINE_AGA:
        results_AGA = selectsensor.select_offline_greedy_lazy_gpu(budget, 20, o_t_approx_kernal2)

    if algorithms == BASELINE_ALL or algorithms == BASELINE_GA:
        results_GA = selectsensor.select_offline_GA(budget, o_t_iter_kernal)
    # plots.save_data(results_GA, 'plot_data_splat/fig1-homo/GA')

    if algorithms == BASELINE_ALL or algorithms == BASELINE_RAN:  # Random
        results_RAN = selectsensor.select_offline_random(budget, 20)
        # plots.save_data(results_RAN, 'plot_data_splat/fig1-homo/random')

    if algorithms == BASELINE_ALL or algorithms == BASELINE_COV:  # Coverage
        results_COV = selectsensor.select_offline_coverage(budget, 20)

    true_x = np.random.choice(range(grid_len), size=100, replace=True)
    true_y = np.random.choice(range(grid_len), size=100, replace=True)
    for j in range(1, budget+1):
        error_AGA = np.zeros(budget + 1)
        error_WAGA = np.zeros(budget + 1)
        error_GA = np.zeros(budget + 1)
        error_COV = np.zeros(budget + 1)
        error_RAN = np.zeros(budget + 1)

        for tno, trans in enumerate(true_x):
            if algorithms == BASELINE_ALL or algorithms == BASELINE_WAGA:
                error_WAGA[j] += selectsensor.compute_conditional_error(trans, true_y[tno],
                                                                    results_WAGA[j-1][3])
            if algorithms == BASELINE_ALL or algorithms == BASELINE_AGA:
                error_AGA[j] += selectsensor.compute_conditional_error(trans, true_y[tno],
                                                                results_AGA[j-1][3])
            if algorithms == BASELINE_ALL or algorithms == BASELINE_GA:
                error_GA[j] += selectsensor.compute_conditional_error(trans, true_y[tno],
                                                                   results_GA[j-1][2])
            if algorithms == BASELINE_ALL or algorithms == BASELINE_COV:
                error_COV[j] += selectsensor.compute_conditional_error(trans, true_y[tno],
                                                                       results_COV[j-1][2])
            if algorithms == BASELINE_ALL or algorithms == BASELINE_RAN:
                error_RAN[j] += selectsensor.compute_conditional_error(trans, true_y[tno],
                                                                   results_RAN[j-1][2])
        error_AGA /= 100
        error_WAGA /= 100
        error_GA /= 100
        error_COV /= 100
        error_RAN /= 100
        #print(j, error_AGA[j], error_WAGA[j], error_GA[j], error_COV[j], error_RAN[j])
        print(j, end=' ')
        if algorithms == BASELINE_ALL or algorithms == BASELINE_WAGA:
            print(error_WAGA[j], end=' ')
        if algorithms == BASELINE_ALL or algorithms == BASELINE_AGA:
            print(error_AGA[j], end=' ')
        if algorithms == BASELINE_ALL or algorithms == BASELINE_GA:
            print(error_GA[j], end=' ')
        if algorithms == BASELINE_ALL or algorithms == BASELINE_RAN:
            #print(cumul_RAN, results_RAN[i][1])
            print(error_RAN[j], end=' ')
        if algorithms == BASELINE_ALL or algorithms == BASELINE_COV:
            print(error_COV[j], end=' ')
        print('')

def test_large():
    cov_file = 'dataSplat/4096/cov'
    sensor_file = 'dataSplat/4096/sensors'
    intruder_hypo_file = 'dataSplat/4096/hypothesis'
    selectsensor = SelectSensor(64)
    selectsensor.init_data(cov_file, sensor_file, intruder_hypo_file)
    budget = 20
    AGA_cpu_end = selectsensor.select_offline_greedy_p_lazy_cpu(budget, 20)

def test_small():
    selectsensor = SelectSensor(10)
    cov_file = 'dataSplat/100/cov1'
    sensor_file = 'dataSplat/100/sensors1'
    intruder_hypo_file = 'dataSplat/100/hypothesis1'
    selectsensor.init_data(cov_file, sensor_file, intruder_hypo_file)
    selectsensor.rescale_intruder_hypothesis()
    selectsensor.transmitters_to_array()  # for GPU
    budget = 10
    AGA_cpu_end = selectsensor.select_offline_greedy_p_lazy_cpu(budget, 10)

def test_standard():
    selectsensor = SelectSensor(40)
    cov_file = 'dataSplat/1600-100/cov'
    sensor_file = 'dataSplat/1600-100/sensors'
    intruder_hypo_file = 'dataSplat/1600-100/hypothesis'
    selectsensor.init_data(cov_file, sensor_file, intruder_hypo_file)
    selectsensor.rescale_intruder_hypothesis()
    selectsensor.transmitters_to_array()  # for GPU
    budget = 20
    AGA_cpu_end = selectsensor.select_offline_greedy_p(budget, 10)


if __name__ == '__main__':
    #test_map()
    # ipsn_homo()
    #test_ipsn_homo(algorithms=BASELINE_ALL)
    #test_weighted_ipsn(algorithms=BASELINE_ALL)
    #test_splat(False, 2)
    #test_splat(False, 3)
    # test_splat(True, 1)
    #test_splat(True, 2)
    #test_splat_localization_single_intruder()
    #select_online_random(self, budget, cores, true_index=-1)
    #test_splat(False, 3)
    #test_utah_baseline(BASELINE_ALL, 20)
    #test_dartmouth_baseline(BASELINE_ALL)
    #test_weighted_utah_baseline(BASELINE_ALL, 10)
    #test_splat_opt()
    #test_splat_baseline(SMALL_INSTANCE, BASELINE_ALL, num_iterations=20)
    
    test_outdoor_baseline(BASELINE_ALL)
    
    #test_outdoor_weighted(BASELINE_ALL)
    #test_weighted_baseline(SMALL_INSTANCE, BASELINE_ALL, num_iterations=1)
    #test_weighted_hetero_baseline(SMALL_INSTANCE, BASELINE_ALL, num_iterations=10)
    #test_splat_hetero_baseline(SMALL_INSTANCE, BASELINE_ALL, num_iterations=20)
    #test_splat_scalability(large=False)
    #test_splat_total_sensors()
    #test_splat_hetero(0)
    #test_approx_ratio(STD_INSTANCE, num_iterations=1)
    # test_small()
    #test_update_hypothesis()
    #test_varying_sensor_density(BASELINE_ALL, 20)
    # test_standard()