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
import line_profiler

#Define constants for flags
BASELINE_ALL = 0
BASELINE_GA = 1
BASELINE_COV = 2
BASELINE_RAN = 3
BASELINE_AGA = 4
LARGE_INSTANCE = True
SMALL_INSTANCE = False
ONLY_INTRUDERS = 1
PRIMARY_INTRUDERS = 2
PRIMARY_SECONDARY_INTRUDERS = 3

def ipsn_homo():
    '''2019 IPSN version
    '''
    selectsensor = SelectSensor(grid_len=50)
    selectsensor.init_data('data50/homogeneous-200/cov', 'data50/homogeneous-200/sensors', 'data50/homogeneous-200/hypothesis')
    # CPU version
    #selectsensor.select_offline_greedy(40)
    #selectsensor.select_offline_greedy2(5)
    #selectsensor.select_offline_greedy_p(20, 16)
    #selectsensor.select_offline_greedy_p_lazy_cpu(20, 16)
    #selectsensor.select_offline_greedy_p_lazy_gpu(1, 12, o_t_approx_kernal)

    # GPU version
    selectsensor.transmitters_to_array()

    #start = time.time()
    #selectsensor.select_offline_greedy_p_lazy_gpu(15, 12, o_t_approx_kernal)
    #print('time = {}'.format(time.time()-start))

    results = selectsensor.select_offline_greedy_lazy_gpu(50, 10, o_t_approx_kernal2)
    for r in results:
        print(r[:-1])

    #plots.figure_1a(selectsensor, None)


def test_ipsn_hetero():
    '''2019 IPSN version
    '''
    selectsensor = SelectSensor('config/ipsn_config.json')
    selectsensor.init_data('data16/heterogeneous/cov', 'data16/heterogeneous/sensors', 'data16/heterogeneous/hypothesis')
    selectsensor.select_offline_greedy_hetero(5, 12, o_t_approx_kernal2)
    selectsensor.select_offline_greedy_hetero(5, 12, o_t_approx_kernal2)


#@profile
def test_splat(large=True, type_of_transmitter=ONLY_INTRUDERS):
    '''2019 Mobicom version using data generated from SPLAT
    Args:
        large (bool): True for 4096 hypothesis, False for 1600 hypothesis
        type_of_transmitter (int):   1 for just intruders, 2 for complete steps, 3 for directly using added hypothesis
    '''
    if large is False:
        config              = 'config/splat_config_40.json'
        cov_file            = 'dataSplat/1600-100/cov'
        sensor_file         = 'dataSplat/1600-100/sensors'
        intruder_hypo_file  = 'dataSplat/1600-100/hypothesis'
        primary_hypo_file   = 'dataSplat/1600-100/hypothesis_primary'
        secondary_hypo_file = 'dataSplat/1600-100/hypothesis_secondary'

        if type_of_transmitter == ONLY_INTRUDERS:      # just the intruders
            selectsensor = SelectSensor(40)
            selectsensor.init_data(cov_file, sensor_file, intruder_hypo_file)
            selectsensor.rescale_intruder_hypothesis()
            selectsensor.transmitters_to_array()
            
            # results_AGA = selectsensor.select_offline_greedy_lazy_gpu(30, 12, o_t_approx_kernal2)

            start = time.time()
            results_GA  = selectsensor.select_offline_GA_old(40, 12)
            print('time = ', time.time()-start, '\n', results_GA)
            
            start = time.time()
            results_GA  = selectsensor.select_offline_GA(40, 12, o_t_iter_kernal)
            print('time = ', time.time()-start, '\n', results_GA)
            
            # plots.save_data_AGA(results, 'plot_data_splat/fig1-homo/AGA')
            # for aga, ga in zip(results_AGA, results_GA):
            #     print(aga, ga)

        if type_of_transmitter == PRIMARY_SECONDARY_INTRUDERS:      # the complete process: intruder --> primary --> secondary
            selectsensor = SelectSensor(40)
            selectsensor.init_data(cov_file, sensor_file, intruder_hypo_file)                # init intruder hypo
            selectsensor.setup_primary_transmitters([123, 1357], primary_hypo_file)          # setup primary
            selectsensor.add_primary(primary_hypo_file)
            selectsensor.setup_secondary_transmitters([456, 789, 1248], secondary_hypo_file) # setup secondary
            selectsensor.add_secondary(secondary_hypo_file)
            selectsensor.rescale_all_hypothesis()
            selectsensor.transmitters_to_array()
            results = selectsensor.select_offline_greedy_lazy_gpu(40, 10, o_t_approx_kernal2)
            for r in results:
                print(r)
        if type_of_transmitter == 3:      # use added hypo directly
            selectsensor = SelectSensor(40)
            selectsensor.init_data(cov_file, sensor_file, intruder_hypo_file)
            selectsensor.rescale_intruder_hypothesis()
            selectsensor.transmitters_to_array()
            results = selectsensor.select_offline_greedy_lazy_gpu(80, 10, o_t_approx_kernal2)
            for r in results:
                print(r)

    else: # large is True
        config              = 'config/splat_config_64.json'
        cov_file            = 'dataSplat/4096/cov'
        sensor_file         = 'dataSplat/4096/sensors'
        intruder_hypo_file  = 'dataSplat/4096/hypothesis'
        primary_hypo_file   = 'dataSplat/4096/hypothesis_primary'
        secondary_hypo_file = 'dataSplat/4096/hypothesis_secondary'

        if type_of_transmitter == ONLY_INTRUDERS:      # just the intruders
            selectsensor = SelectSensor(64)
            selectsensor.init_data(cov_file, sensor_file, intruder_hypo_file)
            selectsensor.rescale_intruder_hypothesis()
            selectsensor.transmitters_to_array()
            results_AGA = selectsensor.select_offline_greedy_lazy_gpu(20, 12, o_t_approx_kernal2)
            results_GA = selectsensor.select_offline_GA_old(20, 20)
            # results_COV = selectsensor.select_offline_coverage(20, 12)
            # results_RAN = selectsensor.select_offline_random(20, 12)
            #plots.save_data(results_AGA, 'plot_data_splat/fig2-homo-small/coverage{}'.format(i))
            #plots.save_data(results_GA, 'plot_data_splat/fig2-homo-small/random{}'.format(i))
            #plots.save_data(results_COV, 'plot_data_splat/fig2-homo-small/random{}'.format(i))
            #plots.save_data(results_RAN, 'plot_data_splat/fig2-homo-small/GA{}'.format(i))

            for j in range(len(results_AGA)):
                print(results_AGA[j], results_GA[j])#, results_COV[j], results_RAN[j])

        if type_of_transmitter == PRIMARY_SECONDARY_INTRUDERS:      # the complete process
            selectsensor = SelectSensor(config)
            selectsensor.init_data(cov_file, sensor_file, intruder_hypo_file)                 # init intruder hypo
            selectsensor.setup_primary_transmitters([479, 3456], primary_hypo_file)           # setup primary
            selectsensor.add_primary(primary_hypo_file)
            selectsensor.setup_secondary_transmitters([789, 1357, 2345], secondary_hypo_file) # setup secondary
            selectsensor.add_secondary(secondary_hypo_file)
            selectsensor.rescale_all_hypothesis()
            selectsensor.transmitters_to_array()
            results = selectsensor.select_offline_greedy_lazy_gpu(20, 10, o_t_approx_kernal2)
            for r in results:
                print(r)
        if type_of_transmitter == 3:      # use added hypo directly
            selectsensor = SelectSensor(config)
            selectsensor.init_data(cov_file, sensor_file, all_hypo_file)
            selectsensor.rescale_intruder_hypothesis()
            selectsensor.transmitters_to_array()
            results = selectsensor.select_offline_greedy_lazy_gpu(20, 10, o_t_approx_kernal2)
            for r in results:
                print(r)


def test_splat_baseline(large, algorithms):
    '''The baseline (GA, random, coverage), without background, homogeneous, 40 x 40 grid
    '''

    if large is False:
        config              = 'config/splat_config_40.json'
        cov_file            = 'dataSplat/1600-100/cov'
        sensor_file         = 'dataSplat/1600-100/sensors'
        intruder_hypo_file  = 'dataSplat/1600-100/hypothesis'
        selectsensor = SelectSensor(40)
        selectsensor.init_data(cov_file, sensor_file, intruder_hypo_file)
        selectsensor.rescale_intruder_hypothesis()
        #selectsensor.transmitters_to_array()        # for GPU
        budget = 30
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

    if algorithms == BASELINE_ALL or algorithms == BASELINE_AGA:
        selectsensor.transmitters_to_array()
        results_AGA = selectsensor.select_offline_greedy_lazy_gpu(budget, 20, o_t_approx_kernal2)
        plots.save_data(results_AGA, 'plot_data_splat/fig1-homo/GA')

    if algorithms == BASELINE_ALL or algorithms == BASELINE_GA:  # GA
        selectsensor.transmitters_to_array()
        results_GA = selectsensor.select_offline_GA_old(budget, 20)
        plots.save_data(results_GA, 'plot_data_splat/fig1-homo/GA')

    if algorithms == BASELINE_ALL or algorithms == BASELINE_RAN:  # Random
        results_RAN = selectsensor.select_offline_random(budget, 20)
        plots.save_data(results_RAN, 'plot_data_splat/fig1-homo/random')

    if algorithms == BASELINE_ALL or algorithms == BASELINE_COV:  # Coverage
        results_COV = selectsensor.select_offline_coverage(budget, 20)
        plots.save_data(results_COV, 'plot_data_splat/fig1-homo/coverage')

    for i in range(len(results_AGA)):
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

def test_splat_opt():
    '''Comparing AGA to the optimal and baselines, without background, homogeneous, small grid 10 x 10
    '''
    config              = 'config/splat_config_10.json'
    cov_file            = 'dataSplat/100/cov{}'
    sensor_file         = 'dataSplat/100/sensors{}'
    intruder_hypo_file  = 'dataSplat/100/hypothesis{}'

    for i in range(1, 2):
        print('\ncase {}'.format(i))
        selectsensor = SelectSensor(10)
        selectsensor.init_data(cov_file.format(i), sensor_file.format(i), intruder_hypo_file.format(i))
        selectsensor.rescale_intruder_hypothesis()
        selectsensor.transmitters_to_array()        # for GPU

        # results = selectsensor.select_offline_greedy_lazy_gpu(10, 12, o_t_approx_kernal2)
        # plots.save_data_AGA(results, 'plot_data_splat/fig2-homo-small/AGA{}'.format(i))

        results = selectsensor.select_offline_GA_old(10, 10)
        plots.save_data(results, 'plot_data_splat/fig2-homo-small/GA{}'.format(i))

        # results = selectsensor.select_offline_GA(10, 10, o_t_iter_kernal)
        
        #results = selectsensor.select_offline_coverage(10, 10)
        #plots.save_data(results, 'plot_data_splat/fig2-homo-small/coverage{}'.format(i))

        #results = selectsensor.select_offline_random(10, 10)
        #plots.save_data(results, 'plot_data_splat/fig2-homo-small/random{}'.format(i))

        # plot_data = []
        # for budget in range(1, 11):
        #     budget, ot = selectsensor.select_offline_optimal(budget, 12)
        #     plot_data.append([budget, ot])
        # plots.save_data(plot_data,'plot_data_splat/fig2-homo-small/optimal{}'.format(i))


def test_splat_total_sensors():
    '''AGA against varies total # of sensors
    '''
    config              = 'config/splat_config_40-50.json'
    cov_file            = 'dataSplat/1600-50/cov'
    sensor_file         = 'dataSplat/1600-50/sensors'
    intruder_hypo_file  = 'dataSplat/1600-50/hypothesis'
    selectsensor = SelectSensor(config)
    selectsensor.init_data(cov_file, sensor_file, intruder_hypo_file)
    selectsensor.rescale_intruder_hypothesis()
    selectsensor.transmitters_to_array()
    results = selectsensor.select_offline_greedy_lazy_gpu(20, 12, o_t_approx_kernal2)
    plots.save_data_AGA(results, 'plot_data_splat/fig4-homo-total-sensors/50-sensors')

    config              = 'config/splat_config_40-100.json'
    cov_file            = 'dataSplat/1600-100/cov'
    sensor_file         = 'dataSplat/1600-100/sensors'
    intruder_hypo_file  = 'dataSplat/1600-100/hypothesis'
    selectsensor = SelectSensor(config)
    selectsensor.init_data(cov_file, sensor_file, intruder_hypo_file)
    selectsensor.rescale_intruder_hypothesis()
    selectsensor.transmitters_to_array()
    results = selectsensor.select_offline_greedy_lazy_gpu(20, 12, o_t_approx_kernal2)
    plots.save_data_AGA(results, 'plot_data_splat/fig4-homo-total-sensors/100-sensors')

    config              = 'config/splat_config_40.json'
    cov_file            = 'dataSplat/1600/cov'
    sensor_file         = 'dataSplat/1600/sensors'
    intruder_hypo_file  = 'dataSplat/1600/hypothesis'
    selectsensor = SelectSensor(config)
    selectsensor.init_data(cov_file, sensor_file, intruder_hypo_file)
    selectsensor.rescale_intruder_hypothesis()
    selectsensor.transmitters_to_array()
    results = selectsensor.select_offline_greedy_lazy_gpu(20, 12, o_t_approx_kernal2)
    plots.save_data_AGA(results, 'plot_data_splat/fig4-homo-total-sensors/200-sensors')

    config              = 'config/splat_config_40-400.json'
    cov_file            = 'dataSplat/1600-400/cov'
    sensor_file         = 'dataSplat/1600-400/sensors'
    intruder_hypo_file  = 'dataSplat/1600-400/hypothesis'
    selectsensor = SelectSensor(config)
    selectsensor.init_data(cov_file, sensor_file, intruder_hypo_file)
    selectsensor.rescale_intruder_hypothesis()
    selectsensor.transmitters_to_array()
    results = selectsensor.select_offline_greedy_lazy_gpu(20, 12, o_t_approx_kernal2)
    plots.save_data_AGA(results, 'plot_data_splat/fig4-homo-total-sensors/400-sensors')

    config              = 'config/splat_config_40-800.json'
    cov_file            = 'dataSplat/1600-800/cov'
    sensor_file         = 'dataSplat/1600-800/sensors'
    intruder_hypo_file  = 'dataSplat/1600-800/hypothesis'
    selectsensor = SelectSensor(config)
    selectsensor.init_data(cov_file, sensor_file, intruder_hypo_file)
    selectsensor.rescale_intruder_hypothesis()
    selectsensor.transmitters_to_array()
    results = selectsensor.select_offline_greedy_lazy_gpu(20, 12, o_t_approx_kernal2)
    plots.save_data_AGA(results, 'plot_data_splat/fig4-homo-total-sensors/800-sensors')


def test_splat_hetero(algorithms):
    '''The baseline (GA, random, coverage), without background, heterogeneous, 40 x 40 grid
    '''
    config              = 'config/splat_config_40.json'
    cov_file            = 'dataSplat/1600-hetero/cov'
    sensor_file         = 'dataSplat/1600-hetero/sensors'
    intruder_hypo_file  = 'dataSplat/1600-hetero/hypothesis-25'
    selectsensor = SelectSensor(config)
    selectsensor.init_data(cov_file, sensor_file, intruder_hypo_file)
    selectsensor.rescale_intruder_hypothesis()
    selectsensor.transmitters_to_array()        # for GPU

    if algorithms == BASELINE_ALL or algorithms == BASELINE_AGA:  # AGA
        results = selectsensor.select_offline_greedy_hetero(30, 12, o_t_approx_kernal2)
        plots.save_data(results, 'plot_data_splat/fig3-hetero/AGA')

    if algorithms == BASELINE_ALL or algorithms == BASELINE_RAN:  # Random
        results = selectsensor.select_offline_random_hetero(50, 12)
        plots.save_data(results, 'plot_data_splat/fig3-hetero/random')

    if algorithms == BASELINE_ALL or algorithms == BASELINE_COV:  # Coverage
        results = selectsensor.select_offline_coverage_hetero(50, 12)
        plots.save_data(results, 'plot_data_splat/fig3-hetero/coverage')

    if algorithms == BASELINE_ALL or algorithms == BASELINE_GA:  # GA
        results = selectsensor.select_offline_GA_hetero(30, 12)
        plots.save_data(results, 'plot_data_splat/fig3-hetero/GA')


def test_ipsn2():
    '''2019 Mobicom version using data generated from IPSN
    '''

    cov_file           = 'dataSplat/1600/cov'
    sensor_file        = 'dataSplat/1600/sensors'
    intruder_hypo_file = 'dataSplat/1600/hypothesis'
    legal_hypo_file    = 'dataSplat/1600/hypothesis_legal'
    add_hypo_file      = 'dataSplat/1600/hypothesis_add'

    selectsensor = SelectSensor('config/splat_config.json')
    #selectsensor.init_data(cov_file, sensor_file, intruder_hypo_file)
    selectsensor.init_data(cov_file, sensor_file, add_hypo_file)
    #selectsensor.setup_legal_transmitters([123, 456, 789, 1357, 1248], legal_hypo_file)
    #selectsensor.add_legal_and_intruder_hypothesis(legal_hypo_file, intruder_hypo_file, add_hypo_file)
    #selectsensor.rescale_add_hypothesis()
    selectsensor.rescale_intruder_hypothesis()
    selectsensor.transmitters_to_array()
    print(selectsensor.select_offline_greedy_lazy_gpu(20, 12, o_t_approx_kernal2))

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



if __name__ == '__main__':
    #test_map()
    # ipsn_homo()
    #test_ipsn_hetero()
    #test_splat(LARGE_INSTANCE, ONLY_INTRUDERS)
    test_splat(SMALL_INSTANCE, ONLY_INTRUDERS)
    #test_splat(False, 2)
    #test_splat(False, 3)
    # test_splat(True, 1)
    #test_splat(True, 2)
    #test_splat_localization_single_intruder()
    #select_online_random(self, budget, cores, true_index=-1)
    #test_splat(False, 3)
    #test_splat_opt()
    # test_splat_baseline(LARGE_INSTANCE, BASELINE_ALL)
    # test_splat_opt()
    #test_splat_total_sensors()
    #test_splat_hetero(0)
