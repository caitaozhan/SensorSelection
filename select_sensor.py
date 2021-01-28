'''
Select sensor and detect transmitter
'''

import random
import math
import copy
import time
import os
import numpy as np
import pandas as pd
from scipy.spatial import distance
from scipy.stats import multivariate_normal, norm
from joblib import Parallel, delayed, dump, load
from sensor import Sensor
from transmitter import Transmitter
from utility import read_config, ordered_insert, power_2_db, power_2_db_, db_2_power, db_2_power_, find_elbow#, print_results
from counter import Counter
try:
    from numba import cuda
    from cuda_kernals import o_t_approx_kernal, o_t_kernal, o_t_approx_dist_kernal, \
                             o_t_approx_kernal2, o_t_approx_dist_kernal2, update_dot_of_selected_kernal, sum_reduce, \
                             o_t_iter_kernal, prod_reduce#, o_t_iter_dist_kernal
except Exception as e:
    pass
import traceback
#from itertools import combinations
#import line_profiler
#from sklearn.cluster import KMeans
#from scipy.optimize import nnls
from scipy.stats import norm
#from plots import visualize_sensor_output, visualize_cluster, visualize_localization, visualize_q_prime, visualize_q, visualize_splot, visualize_unused_sensors
#from utility import generate_intruders, generate_intruders_2
#from skimage.feature import peak_local_max
#import itertools

RTL_SDR_NOISE_FLOOR = -80
WIFI_NOISE_FLOOR = -50

def q_lookup_array(q_params, lookup_table):
    q_params[q_params > 8.3] = 8.3 - 0.0001
    index = (q_params * 10000).astype(int)
    #print(index)
    return lookup_table[index]

def q_lookup(q_param, lookup_table):
    '''The lookup table, devide range [0, 8.3], the bin size is 0.0001
       The definition of this lookup table is given at the __init__ function of class SelectSensor
    Params:
        q_param (flaot):
        lookup_table (np.ndarray, n=1):
    Return:
        (float)
    '''
    if q_param > 8.3:
        return 0.
    index = int(q_param*10000)
    return lookup_table[index]

class SelectSensor:
    '''Near-optimal low-cost sensor selection

    Attributes:
        config (json):               configurations - settings and parameters
        sen_num (int):               the number of sensors
        grid_len (int):              the length of the grid
        grid_priori (np.ndarray):    the element is priori probability of hypothesis - transmitter
        grid_posterior (np.ndarray): the element is posterior probability of hypothesis - transmitter
        transmitters (list):         a list of Transmitter
        sensors (dict):              a dictionary of Sensor. less than 10% the # of transmitter
        data (ndarray):              a 2D array of observation data
        covariance (np.ndarray):     a 2D array of covariance. each data share a same covariance matrix
        mean_stds (dict):            assume sigal between a transmitter-sensor pair is normal distributed
        subset (dict):               a subset of all sensors
        subset_index (list):         the linear index of sensor in self.sensors
        meanvec_array (np.ndarray):  contains the mean vector of every transmitter, for CUDA
        TPB (int):                   thread per block
        legal_transmitter (list):    a list of legal transmitters
        lookup_table (np.array):     trade space for time on the q function
    '''
    def __init__(self, grid_len, debug=False):
        self.grid_len = grid_len
        self.sen_num  = 0
        self.grid_priori = np.zeros(0)
        self.grid_posterior = np.zeros(0)
        self.transmitters = []                 # transmitters are the hypothesises
        self.intruders = []
        self.sensors = []
        self.sensors_used = np.array(0)
        self.sensors_collect = {}              # precomputed collected sensors
        self.key = '{}-{}'                     # key template for self.sensors_collect
        self.data = np.zeros(0)
        self.covariance = np.zeros(0)
        self.init_transmitters()
        self.set_priori()
        self.means = np.zeros(0)               # negative mean of intruder
        self.means_primary = np.zeros(0)       # negative mean of intruder plus primary
        self.means_all = np.zeros(0)           # negative mean of intruder plus primary plus secondary (all)
        self.means_rescale = np.zeros(0)       # positive mean of either self.means or self.means_rescale
        self.stds = np.zeros(0)                # for tx, sensor pair
        self.subset = {}
        self.subset_index = []
        self.meanvec_array = np.zeros(0)
        self.TPB = 32
        self.primary_trans = []                # introduce the legal transmitters as secondary user in the Mobicom version
        self.secondary_trans = []              # they include primary and secondary
        self.lookup_table_q = np.array([1. - 0.5*(1. + math.erf(i/1.4142135623730951)) for i in np.arange(0, 8.3, 0.0001)])
        #self.lookup_table_norm = norm(0, 1).pdf(np.arange(0, 39, 0.0001))  # norm(0, 1).pdf(39) = 0
        self.counter = Counter()               # timer
        self.debug  = debug                    # debug mode do visulization stuff, which is time expensive
        self.present = np.zeros(self.grid_len * self.grid_len)

    #@profile
    def init_data(self, cov_file, sensor_file, hypothesis_file):
        '''Init everything from collected real data
           1. init covariance matrix
           2. init sensors
           3. init mean and std between every pair of transmitters and sensors
        '''
        cov = pd.read_csv(cov_file, header=None, delimiter=' ')
        #del cov[len(cov)]
        self.covariance = cov.values

        self.sensors = []
        with open(sensor_file, 'r') as f:
            max_gain = 0.5*len(self.transmitters)
            index = 0
            lines = f.readlines()
            for line in lines:
                line = line.split(' ')
                x, y, std, cost = int(line[0]), int(line[1]), float(line[2]), float(line[3])
                self.sensors.append(Sensor(x, y, std, cost, gain_up_bound=max_gain, index=index))  # uniform sensors
                index += 1
        self.sen_num = len(self.sensors)

        self.means = np.zeros((self.grid_len * self.grid_len, len(self.sensors)))
        self.stds = np.zeros((self.grid_len * self.grid_len, len(self.sensors)))
        #self.grid_priori = np.zeros((self.grid_len, self.grid_len))
        with open(hypothesis_file, 'r') as f:
            lines = f.readlines()
            count = 0
            for line in lines:
                try:
                    line = line.split(' ')
                    tran_x, tran_y = int(line[0]), int(line[1])
                    #sen_x, sen_y = int(line[2]), int(line[3])
                    mean, std = float(line[4]), float(line[5])
                except:
                    print('Exception', line)
                self.means[tran_x*self.grid_len + tran_y, count] = mean  # count equals to the index of the sensors
                self.stds[tran_x*self.grid_len + tran_y, count] = std
                #self.grid_priori[tran_x, tran_y] = 1
                self.present[tran_x * self.grid_len + tran_y] = 1
                count = (count + 1) % len(self.sensors)
        denominator = np.sum(self.grid_priori)
        #self.grid_priori /= denominator

        #temp_mean = np.zeros(self.grid_len * self.grid_len, )
        for transmitter in self.transmitters:
            tran_x, tran_y = transmitter.x, transmitter.y
            mean_vec = [0] * len(self.sensors)
            for sensor in self.sensors:
                mean = self.means[self.grid_len*tran_x + tran_y, sensor.index]
                mean_vec[sensor.index] = mean
            transmitter.mean_vec = np.array(mean_vec)


    def vary_power(self, powers):
        '''Varing power
        Args:
            powers (list): an element is a number that denote the difference from the default power read from the hypothesis file
        '''
        for tran in self.transmitters:
            tran.powers = powers


    def interpolate_gradient(self, x, y):
        '''The gradient for location (x, y) in the origin grid, for each sensor
        Args:
            x (int)
            y (int)
        Return:
            grad_x, grad_y (np.array, np.array): gradients in the x and y direction
        '''
        grad_x = np.zeros(len(self.sensors))
        grad_y = np.zeros(len(self.sensors))
        for s_index in range(len(self.sensors)):
            if x + 1 < self.grid_len:
                origin1 = self.means[x*self.grid_len + y][s_index]
                origin2 = self.means[(x+1)*self.grid_len + y][s_index]
                grad_x[s_index] = origin2 - origin1
            else:
                origin1 = self.means[x*self.grid_len + y][s_index]
                origin2 = self.means[(x-1)*self.grid_len + y][s_index]
                grad_x[s_index] = origin1 - origin2
            if y + 1 < self.grid_len:
                origin1 = self.means[x*self.grid_len + y][s_index]
                origin2 = self.means[x*self.grid_len + y+1][s_index]
                grad_y[s_index] = origin2 - origin1
            else:
                origin1 = self.means[x*self.grid_len + y][s_index]
                origin2 = self.means[x*self.grid_len + y-1][s_index]
                grad_y[s_index] = origin1 - origin2
        return grad_x, grad_y


    def interpolate_loc(self, scale, hypo_file, sensor_file):
        '''From M hypothesis to scale^2 * M hypothesis. For localization. Don't change the origin class members, instead create new copies.
           1. self.means_loc
           2. self.transmitters_loc
           3. self.sensors_loc
           4. self.grid_len_loc

        Args:
            scale (int): scaling factor
            hypo_file (str):   need to expand the hypothesis file
            sensor_file (str): need to change the coordinate of each sensor
        '''
        self.grid_len_loc = scale*self.grid_len
        self.means_loc = np.zeros((self.grid_len_loc * self.grid_len_loc, len(self.sensors)))
        self.transmitters_loc = [0] * self.grid_len_loc * self.grid_len_loc
        self.sensors_loc = copy.deepcopy(self.sensors)

        for t_index in range(len(self.transmitters_loc)):
            i = t_index // self.grid_len_loc
            j = t_index %  self.grid_len_loc
            self.transmitters_loc[t_index] = Transmitter(i, j)
        for s_index in range(len(self.sensors_loc)):
            self.sensors_loc[s_index].x = scale*self.sensors[s_index].x
            self.sensors_loc[s_index].y = scale*self.sensors[s_index].y

        for t_index in range(len(self.transmitters)):                          # M
            x = self.transmitters[t_index].x
            y = self.transmitters[t_index].y
            grad_x, grad_y = self.interpolate_gradient(x, y)
            x_loc = scale*x
            y_loc = scale*y
            for i in range(scale):
                for j in range(scale):                                         # scale^2
                    new_t_index = (x_loc+i)*self.grid_len_loc + (y_loc+j)
                    for s_index in range(len(self.sensors)):                   # S
                        origin_rss = self.means[t_index][s_index]              # = O(M*scale^2*S)
                        interpolate = origin_rss + float(i)/scale*grad_x[s_index] + float(j)/scale*grad_y[s_index]
                        self.means_loc[new_t_index][s_index] = interpolate
        
        with open(hypo_file, 'w') as f:
            for t_index in range(len(self.transmitters_loc)):
                trans_x = t_index // self.grid_len_loc
                trans_y = t_index % self.grid_len_loc
                for s_index in range(len(self.sensors_loc)):
                    sen_x = self.sensors_loc[s_index].x
                    sen_y = self.sensors_loc[s_index].y
                    mean = self.means_loc[t_index][s_index]
                    std   = self.sensors_loc[s_index].std
                    f.write('{} {} {} {} {} {}\n'.format(trans_x, trans_y, sen_x, sen_y, mean, std))
       
        with open(sensor_file, 'w') as f:
            for sensor in self.sensors:
                f.write('{} {} {} {}\n'.format(scale*sensor.x, scale*sensor.y, sensor.std, sensor.cost))



    def init_data_from_model(self, num_sensors):
        self.covariance = np.zeros((num_sensors, num_sensors))
        np.fill_diagonal(self.covariance, 1)
        max_gain = 0.5 * len(self.transmitters)

        #diff = (self.grid_len * self.grid_len) // num_sensors
        diff = 3
        count = 0
        for i in range(0, self.grid_len * self.grid_len, diff):
            x = i // self.grid_len
            y = i % self.grid_len
            self.sensors.append(Sensor(x, y, 1, 1, gain_up_bound=max_gain, index=count))
            count += 1

        self.means = np.zeros((self.grid_len * self.grid_len, len(self.sensors)))
        self.stds = np.ones((self.grid_len * self.grid_len, len(self.sensors)))

        for i in range(0, self.grid_len):
            for j in range(0, self.grid_len):
                for sensor in self.sensors:
                    distance = np.sqrt((i - sensor.x) ** 2 + (j - sensor.y) ** 2)
                    #print('distance = ', distance)
                    self.means[i * self.grid_len + j, sensor.index] = 10 - distance * 5
                    self.stds[i * self.grid_len + j, sensor.index] = 1
        print(self.means)
        for transmitter in self.transmitters:
            tran_x, tran_y = transmitter.x, transmitter.y
            mean_vec = [0] * len(self.sensors)
            for sensor in self.sensors:
                mean = self.means[self.grid_len * tran_x + tran_y, sensor.index]
                mean_vec[sensor.index] = mean
            transmitter.mean_vec = np.array(mean_vec)

        # del self.means
        # del self.stds
        print('\ninit done!')

    def set_priori(self):
        '''Set priori distribution - uniform distribution
        '''
        uniform = 1./(self.grid_len * self.grid_len)
        self.grid_priori = np.full((self.grid_len, self.grid_len), uniform)
        self.grid_posterior = np.full((self.grid_len, self.grid_len), uniform)
    

    def init_transmitters(self):
        '''Initiate a transmitter at all locations
        '''
        self.transmitters = [0] * self.grid_len * self.grid_len
        for i in range(self.grid_len):
            for j in range(self.grid_len):
                transmitter = Transmitter(i, j)
                setattr(transmitter, 'hypothesis', i*self.grid_len + j)
                self.transmitters[i*self.grid_len + j] = transmitter


    def setup_primary_transmitters(self, primary_transmitter, primary_hypo_file):
        '''Setup the primary transmitters, then "train" the distribution of them by linearly adding up the milliwatt power
        Args:
            primary_transmitter (list): index of legal primary transmitters
            primary_hypo_file (str):    filename, primary RSS to each sensor
        '''
        print('Setting up primary transmitters...', end=' ')
        self.primary_trans = []
        for trans in primary_transmitter:
            x = self.transmitters[trans].x
            y = self.transmitters[trans].y
            self.primary_trans.append(Transmitter(x, y))

        dic_mean = {}   # (sensor.x, sensor.y) --> [legal_mean1, legal_mean2, ...]
        dic_std  = {}   # (sensor.x, sensor.y) --> []
        for sensor in self.sensors:
            dic_mean[(sensor.x, sensor.y)] = []
            dic_std[(sensor.x, sensor.y)] = sensor.std
            for primary in self.primary_trans:
                pri_index = self.grid_len * primary.x + primary.y
                dic_mean[(sensor.x, sensor.y)].append(self.means[pri_index, sensor.index])

        # y = 20*log10(x)
        # x = 10^(y/20)
        # where y is power in dB and x is the absolute value of iq samples, i.e. amplitude
        # do a addition in the absolute value of iq samples
        with open(primary_hypo_file, 'w') as f:
            for key, value in dic_mean.items():
                amplitudes = np.power(10, np.array(value)/20)
                addition = sum(amplitudes)
                power_db = 20*np.log10(addition)
                f.write('{} {} {} {}\n'.format(key[0], key[1], power_db, dic_std[key]))


    def add_primary(self, primary_hypo_file):
        '''Add primary's RSS to itruder's RSS and save the sum to self.means_primary, write to file optional
        Params:
            primary_hypo_file (str):   filename, primary RSS to each each sensor
            intru_pri_hypo_file (str): filename, primary RSS plus intruder RSS to each each sensor
            write (bool): if True write the sums to intru_pri_hypo_file, if False don't write
        '''
        print('Adding primary...')
        hypothesis_legal = {}
        with open(primary_hypo_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.split(' ')
                sen_x = int(line[0])
                sen_y = int(line[1])
                mean  = float(line[2])
                hypothesis_legal[(sen_x, sen_y)] = db_2_power(mean)

        self.means_primary = np.zeros((len(self.transmitters), len(self.sensors)))
        means_amplitute = db_2_power(self.means)
        for trans_index in range(len(self.transmitters)):
            new_means = np.zeros(len(self.sensors))
            for sen_index in range(len(self.sensors)):
                intru_pri_amplitute = means_amplitute[trans_index, sen_index]
                sen_x = self.sensors[sen_index].x
                sen_y = self.sensors[sen_index].y
                lagel_amplitude = hypothesis_legal.get((sen_x, sen_y))
                add_amplitude   = intru_pri_amplitute + lagel_amplitude
                new_means[sen_index] = add_amplitude
            self.means_primary[trans_index, :] = new_means
        self.means_primary = power_2_db(self.means_primary)


    def setup_secondary_transmitters(self, secondary_transmitter, secondary_hypo_file):
        '''Setup the secondary transmitters, then "train" the distribution of them by linearly adding up the milliwatt power
        Args:
            secondary_transmitter (list): index of legal secondary transmitters
            secondary_hypo_file (str):    filename, secondary RSS to each sensor
        '''
        print('Setting up secondary transmitters...', end=' ')
        self.secondary_trans = []                 # a mistake here, forgot to empty it
        for trans in secondary_transmitter:
            x = self.transmitters[trans].x
            y = self.transmitters[trans].y
            self.secondary_trans.append(Transmitter(x, y))

        dic_mean = {}   # (sensor.x, sensor.y) --> [legal_mean1, legal_mean2, ...]
        dic_std  = {}   # (sensor.x, sensor.y) --> []
        for sensor in self.sensors:
            dic_mean[(sensor.x, sensor.y)] = []
            dic_std[(sensor.x, sensor.y)] = sensor.std
            for secondary in self.secondary_trans:
                sec_index = self.grid_len * secondary.x + secondary.y
                dic_mean[(sensor.x, sensor.y)].append(self.means[sec_index, sensor.index])

        # y = 20*log10(x)
        # x = 10^(y/20)
        # where y is power in dB and x is the absolute value of iq samples, i.e. amplitude
        # do a addition in the absolute value of iq samples
        with open(secondary_hypo_file, 'w') as f:
            for key, value in dic_mean.items():
                amplitudes = np.power(10, np.array(value)/20)
                addition = sum(amplitudes)
                power_db = 20*np.log10(addition)
                f.write('{} {} {} {}\n'.format(key[0], key[1], power_db, dic_std[key]))


    #@profile
    def add_secondary(self, secondary_file):
        '''Add secondary's RSS to (itruder's plus primary's) RSS and save the sum to self.means_all, write to file optional
        Params:
            secondary_file (str): filename, secondary RSS to each each sensor
            all_file (str):       filename, intruder RSS + primary RSS + secondary RSS to each each sensor
            write (bool):         if True write the sums to all_file, if False don't write
        '''
        print('Adding secondary...')
        hypothesis_legal = {}
        with open(secondary_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.split(' ')
                sen_x = int(line[0])
                sen_y = int(line[1])
                mean  = float(line[2])
                hypothesis_legal[(sen_x, sen_y)] = db_2_power(mean)

        self.means_all = np.zeros((len(self.transmitters), len(self.sensors)))
        means_primary_amplitute = db_2_power(self.means_primary)
        for trans_index in range(len(self.transmitters)):
            new_means = np.zeros(len(self.sensors))
            for sen_index in range(len(self.sensors)):
                intru_pri_amplitute = means_primary_amplitute[trans_index, sen_index]
                sen_x = self.sensors[sen_index].x
                sen_y = self.sensors[sen_index].y
                lagel_amplitude = hypothesis_legal.get((sen_x, sen_y))
                add_amplitude   = intru_pri_amplitute + lagel_amplitude
                new_means[sen_index] = add_amplitude
            self.means_all[trans_index, :] = new_means
        self.means_all = power_2_db(self.means_all)

    def rescale_wifi_hypothesis(self):
        threshold = -125
        num_trans = len(self.transmitters)
        num_sensors = len(self.sensors)
        self.means_rescale = np.zeros((num_trans, num_sensors))
        for i in range(num_trans):
            for j in range(num_sensors):
                mean = self.means[i, j]
                mean -= threshold
                mean = mean if mean>=0 else 0
                self.means_rescale[i, j] = mean
                self.transmitters[i].mean_vec[j] = mean

    def rescale_intruder_hypothesis(self, noise_floor = RTL_SDR_NOISE_FLOOR, offset=0):
        '''Rescale hypothesis, and save it in a new np.array
        '''
        threshold = noise_floor
        num_trans = len(self.transmitters)
        num_sen   = len(self.sensors)
        self.means_rescale = np.zeros((num_trans, num_sen))
        for i in range(num_trans):
            for j in range(num_sen):
                mean = self.means[i, j] + offset
                mean -= threshold
                mean = mean if mean>=0 else 0
                self.means_rescale[i, j] = mean
                self.transmitters[i].mean_vec[j] = mean


    def rescale_all_hypothesis(self):
        '''Rescale hypothesis, and save it in a new np.array # TODO
        '''
        threshold = -80
        num_trans = len(self.transmitters)
        num_sen   = len(self.sensors)
        self.means_rescale = np.zeros((num_trans, num_sen))
        for i in range(num_trans):
            for j in range(num_sen):
                mean = self.means_all[i, j]
                mean -= threshold
                mean = mean if mean>=0 else 0
                self.means_rescale[i, j] = mean
                self.transmitters[i].mean_vec[j] = mean


    def update_subset(self, subset_index):
        '''Given a list of sensor indexes, which represents a subset of sensors, update self.subset
        Args:
            subset_index (list): a list of sensor indexes. guarantee sorted
        '''
        self.subset = []
        self.subset_index = subset_index
        for index in self.subset_index:
            self.subset.append(self.sensors[index])


    def update_transmitters(self):
        '''Given a subset of sensors' index,
           update each transmitter's mean vector sub and multivariate gaussian function
        '''
        for transmitter in self.transmitters:
            transmitter.set_mean_vec_sub(self.subset_index)
            new_cov = self.covariance[np.ix_(self.subset_index, self.subset_index)]
            transmitter.multivariant_gaussian = multivariate_normal(mean=transmitter.mean_vec_sub, cov=new_cov)


    def update_transmitters_loc(self):
        '''Given a subset of sensors' index,
           update each transmitter's mean vector sub and multivariate gaussian function
        '''
        for transmitter in self.transmitters_loc:
            transmitter.set_mean_vec_sub(self.subset_index)
            new_cov = self.covariance[np.ix_(self.subset_index, self.subset_index)]
            transmitter.multivariant_gaussian = multivariate_normal(mean=transmitter.mean_vec_sub, cov=new_cov)


    def update_mean_vec_sub(self, subset_index):
        '''Given a subset of sensors' index,
           update each transmitter's mean vector sub
        Args:
            subset_index (list)
        '''
        for transmitter in self.transmitters:
            transmitter.set_mean_vec_sub(subset_index)

    def update_mean_vec_sub_loc(self, subset_index):
        for transmitter in self.transmitters_loc:
            transmitter.set_mean_vec_sub(subset_index)

    def covariance_sub(self, subset_index):
        '''Given a list of index of sensors, return the sub covariance matrix
        Args:
            subset_index (list): list of index of sensors. should be sorted.
        Return:
            (np.ndarray): a 2D sub covariance matrix
        '''
        if subset_index is list:
            sub_cov = self.covariance[np.ix_(subset_index, subset_index)]
        else:
            #print(subset_index)
            sub_cov = self.covariance[subset_index, :]
            sub_cov = sub_cov[:, subset_index]
        return sub_cov

    #@profile
    def o_t_cpu(self, subset_index):
        '''Given a subset of sensors T, compute the O_T
        Args:
            subset_index (list): a subset of sensors T, guarantee sorted
        Return O_T
        '''
        if not subset_index:  # empty sequence are false
            return 0
        sub_cov = self.covariance_sub(subset_index)
        sub_cov_inv = np.linalg.inv(sub_cov)        # inverse
        o_t = 0
        for i in range(len(self.transmitters)):
            transmitter_i = self.transmitters[i]
            i_x, i_y = transmitter_i.x, transmitter_i.y
            transmitter_i.set_mean_vec_sub(subset_index)
            prob_i = 1
            for j in range(len(self.transmitters)):
                transmitter_j = self.transmitters[j]
                j_x, j_y = transmitter_j.x, transmitter_j.y
                if i_x == j_x and i_y == j_y:
                    continue
                transmitter_j.set_mean_vec_sub(subset_index)
                pj_pi = transmitter_j.mean_vec_sub - transmitter_i.mean_vec_sub
                prob_i *= (1 - norm.sf(0.5 * math.sqrt(np.dot(np.dot(pj_pi, sub_cov_inv), pj_pi))))
            o_t += prob_i * self.grid_priori[i_x][i_y]
        return o_t


    def o_t_approximate(self, subset_index):
        '''Not the accurate O_T, but apprioximating O_T. So that we have a good propertiy of submodular
        Args:
            subset_index (list): a subset of sensors T, needs guarantee sorted
        Return:
            (float): the approximation of O_acc
        '''
        if not subset_index:  # empty sequence are false
            return -99999999999.
        sub_cov = self.covariance_sub(subset_index)
        sub_cov_inv = np.linalg.inv(sub_cov)         # inverse
        prob_error = 0

        for i in range(len(self.transmitters)):
            transmitter_i = self.transmitters[i]     # when the ground truth is at location i
            i_x, i_y = transmitter_i.x, transmitter_i.y
            transmitter_i.set_mean_vec_sub(subset_index)
            prob_i = 0
            for j in range(len(self.transmitters)):
                transmitter_j = self.transmitters[j] # when the classification is at location j
                j_x, j_y = transmitter_j.x, transmitter_j.y
                if i_x == j_x and i_y == j_y:        # correct classification, so no error
                    continue
                transmitter_j.set_mean_vec_sub(subset_index)
                pj_pi = transmitter_j.mean_vec_sub - transmitter_i.mean_vec_sub
                prob_i += norm.sf(0.5 * math.sqrt(np.dot(np.dot(pj_pi, sub_cov_inv), pj_pi)))
            prob_error += prob_i * self.grid_priori[i_x][i_y]
        return 1 - prob_error


    def o_t_approximate2(self, dot_of_selected, candidate):
        '''Not the accurate O_T, but apprioximating O_T. So that we have a good propertiy of submodular
        Args:
            dot_of_selected (np.ndarray): stores the np.dot(np.dot(pj_pi, sub_cov_inv), pj_pi)) of sensors already selected
                                          in previous iterations. shape=(m, m) where m is the number of hypothesis (grid_len^2)
            candidate (int):              a new candidate
        Return:
            (float): the approximation of O_acc
        '''
        prob_error = 0
        for i in range(len(self.transmitters)):
            transmitter_i = self.transmitters[i]     # when the ground truth is at location i
            i_x, i_y = transmitter_i.x, transmitter_i.y
            prob_i = 0
            for j in range(len(self.transmitters)):
                transmitter_j = self.transmitters[j] # when the classification is at location j
                j_x, j_y = transmitter_j.x, transmitter_j.y
                if i_x == j_x and i_y == j_y:        # correct classification, so no error
                    continue

                dot_of_candidate = ((transmitter_j.mean_vec[candidate] - transmitter_i.mean_vec[candidate]) ** 2) / self.covariance[candidate][candidate]
                dot_of_new_subset = dot_of_selected[i][j] + dot_of_candidate
                prob_i += q_lookup(0.5 * math.sqrt(dot_of_new_subset), self.lookup_table_q)

            prob_error += prob_i * self.grid_priori[i_x][i_y]
        return 1 - prob_error

    def o_t_approximate3(self, dot_of_selected, candidate):
        '''Not the accurate O_T, but apprioximating O_T. So that we have a good propertiy of submodular
        Args:
            dot_of_selected (np.ndarray): stores the np.dot(np.dot(pj_pi, sub_cov_inv), pj_pi)) of sensors already selected
                                          in previous iterations. shape=(m, m) where m is the number of hypothesis (grid_len^2)
            candidate (int):              a new candidate
        Return:
            (float): the approximation of O_acc
        '''
        mean_vecs = [transmitter_i.mean_vec[candidate] for transmitter_i in self.transmitters]
        mean_pairs = np.array(np.meshgrid(mean_vecs, mean_vecs)).T.reshape(2, -1)
        #print(mean_pairs.shape)
        diff_mean = np.square(mean_pairs[1, :] - mean_pairs[0, :])
        diff_mean = diff_mean.reshape(len(self.transmitters), -1)
        dots_of_candidate = diff_mean / self.covariance[candidate][candidate]
        new_dots = dot_of_selected + dots_of_candidate
        #print(new_dots)
        probs_i = q_lookup_array(0.5 * np.sqrt(new_dots), self.lookup_table_q)
        probs_i = np.sum(probs_i, axis=1).reshape(-1, self.grid_len)
        prob_error = np.multiply(probs_i, self.grid_priori)
        prob_error = np.sum(prob_error)
        return 1 - prob_error

    def update_dot_of_selected(self, dot_of_selected, best_candidate):
        '''Update dot_of_selected after a new sensor is seleted
        Args:
            dot_of_selected (np.ndarray): stores the np.dot(np.dot(pj_pi, sub_cov_inv), pj_pi)) of sensors already selected
                                          in previous iterations. shape=(m, m) where m is the number of hypothesis (grid_len^2)
            best_candidate (int):         the best candidate just selected
        '''
        for i in range(len(self.transmitters)):
            transmitter_i = self.transmitters[i]     # when the ground truth is at location i
            i_x, i_y = transmitter_i.x, transmitter_i.y
            for j in range(len(self.transmitters)):
                transmitter_j = self.transmitters[j] # when the classification is at location j
                j_x, j_y = transmitter_j.x, transmitter_j.y
                if i_x == j_x and i_y == j_y:        # correct classification, so no error
                    continue

                dot_of_best_candidate = ((transmitter_j.mean_vec[best_candidate] - transmitter_i.mean_vec[best_candidate]) ** 2) / self.covariance[best_candidate][best_candidate]
                dot_of_selected[i][j] += dot_of_best_candidate


    def select_offline_greedy(self, budget):
        '''Select a subset of sensors greedily. offline + homo version
           The O(BS M^2 B^2) version
        Args:
            budget (int): budget constraint
        Return:
            (list): an element is [str, int, float],
                    where str is the list of subset_index, int is # of sensors, float is O_T
        '''
        cost = 0                                            # |T| in the paper
        subset_index = []                                   # T   in the paper
        complement_index = [i for i in range(self.sen_num)] # S\T in the paper
        plot_data = []

        while cost < budget and complement_index:
            maximum = -9999999999                           # L in the paper
            best_candidate = complement_index[0]            # init the best candidate as the first one
            start = time.time()
            for candidate in complement_index:
                ordered_insert(subset_index, candidate)     # guarantee subset_index always be sorted here
                temp = self.o_t_approximate(subset_index)
                #print(subset_index, temp)
                if temp > maximum:
                    maximum = temp
                    best_candidate = candidate
                subset_index.remove(candidate)
            ordered_insert(subset_index, best_candidate)    # guarantee subset_index always be sorted here
            print('cost = {}, time = {}, o_t = {}'.format(cost+1, time.time()-start, maximum))
            complement_index.remove(best_candidate)
            plot_data.append([str(subset_index), len(subset_index), maximum])
            cost += 1

        return plot_data


    def select_offline_greedy2(self, budget):
        '''Select a subset of sensors greedily. offline + homo version
           The O(BS M^2) version
        Args:
            budget (int): budget constraint
        Return:
            (list): an element is [str, int, float],
                    where str is the list of subset_index, int is # of sensors, float is O_T
        '''
        cost = 0                                            # |T| in the paper
        subset_index = []                                   # T   in the paper
        complement_index = [i for i in range(self.sen_num)] # S\T in the paper
        plot_data = []
        dot_of_selected = np.zeros((len(self.transmitters), len(self.transmitters)))

        while cost < budget and complement_index:
            maximum = -9999999999
            best_candidate = complement_index[0]            # init the best candidate as the first one
            start = time.time()
            for candidate in complement_index:
                temp = self.o_t_approximate2(dot_of_selected, candidate)
                #print(subset_index, temp)
                if temp > maximum:
                    maximum = temp
                    best_candidate = candidate
            self.update_dot_of_selected(dot_of_selected, best_candidate)
            print('cost = {}, time = {}, o_t = {}'.format(cost+1, time.time()-start, maximum))
            ordered_insert(subset_index, best_candidate)    # guarantee subset_index always be sorted here
            complement_index.remove(best_candidate)
            plot_data.append([str(subset_index), len(subset_index), maximum])
            cost += 1

        return plot_data


    def select_offline_greedy_p(self, budget, cores):
        '''(Parallel version) Select a subset of sensors greedily. offline + homo version using ** CPU **
           The O(BS M^2) version
        Args:
            budget (int): budget constraint
            cores (int):  number of cores for parallelzation
        Return:
            (list): an element is [str, int, float],
                    where str is the list of subset_index, int is # of sensors, float is O_T
        '''
        plot_data = []
        cost = 0                                            # |T| in the paper
        subset_index = []                                   # T   in the paper
        complement_index = [i for i in range(self.sen_num)] # S\T in the paper
        subset_to_compute = []
        dot_of_selected = np.zeros((len(self.transmitters), len(self.transmitters)))

        while cost < budget and complement_index:
            start = time.time()
            candidate_result = Parallel(n_jobs=cores)(delayed(self.o_t_approximate3)(dot_of_selected, candidate) for candidate in complement_index)

            best_candidate = complement_index[0]
            maximum = candidate_result[0]
            for i in range(len(candidate_result)):
                #print(complement_index[i], candidate_result[i])
                if candidate_result[i] > maximum:
                    maximum = candidate_result[i]
                    best_candidate = complement_index[i]
            self.update_dot_of_selected(dot_of_selected, best_candidate)
            print('cost = {}, # of batch = {}, time = {}, best = {}, o_t = {}'.format(cost+1, math.ceil(len(complement_index)/cores), time.time() - start, best_candidate, maximum))

            ordered_insert(subset_index, best_candidate)    # guarantee subset_index always be sorted here
            complement_index.remove(best_candidate)
            cost += 1
            subset_to_compute.append(copy.deepcopy(subset_index))
            plot_data.append([len(subset_index), maximum, 0]) # don't compute real o_t now, delay to after all the subsets are selected

            if maximum > 0.999999999:
                break

        subset_results = Parallel(n_jobs=cores)(delayed(self.o_t_cpu)(subset_index) for subset_index in subset_to_compute)

        for i in range(len(subset_results)):
            plot_data[i][2] = subset_results[i]
        return plot_data


    def select_offline_greedy_p_lazy_cpu(self, budget, cores):
        '''(Parallel + Lazy greedy) Select a subset of sensors greedily. offline + homo version using ** CPU **
           The O(BS M^2) version
        Attributes:
            budget (int): budget constraint
            cores (int): number of cores for parallelzation
        Return:
            (list): an element is [str, int, float],
                    where str is the list of subset_index, int is # of sensors, float is O_T
        '''
        counter = 0
        base_ot_approx = 1 - 0.5*len(self.transmitters)
        plot_data = []
        cost = 0                                            # |T| in the paper
        subset_index = []                                   # T   in the paper
        complement_sensors = copy.deepcopy(self.sensors)    # S\T in the paper
        subset_to_compute = []
        dot_of_selected = np.zeros((len(self.transmitters), len(self.transmitters)))

        while cost < budget and complement_sensors:
            best_candidate = -1
            best_sensor = None
            complement_sensors.sort()   # sorting the gain descendingly
            new_base_ot_approx = 0
            update, max_gain = 0, 0
            start = time.time()
            while update < len(complement_sensors):
                update_end = update+cores if update+cores <= len(complement_sensors) else len(complement_sensors)
                candidiate_index = []
                for i in range(update, update_end):
                    candidiate_index.append(complement_sensors[i].index)
                counter += 1

                candidate_results = Parallel(n_jobs=cores)(delayed(self.o_t_approximate3)(dot_of_selected, candidate) for candidate in candidiate_index)

                for i, j in zip(range(update, update_end), range(0, cores)):  # the two range might be different, if the case, follow the first range
                    complement_sensors[i].gain_up_bound = candidate_results[j] - base_ot_approx  # update the upper bound of gain
                    if complement_sensors[i].gain_up_bound > max_gain:
                        max_gain = complement_sensors[i].gain_up_bound
                        best_candidate = candidiate_index[j]
                        best_sensor = complement_sensors[i]
                        new_base_ot_approx = candidate_results[j]

                if update_end < len(complement_sensors) and max_gain > complement_sensors[update_end].gain_up_bound:   # where the lazy happens
                    print('\n***LAZY!***\n', cost, (update, update_end), len(complement_sensors), best_candidate, '\n')
                    break
                update += cores
            self.update_dot_of_selected(dot_of_selected, best_candidate)
            #print('cost = {}, time = {}, best = {}, o_t = {}'.format(cost+1, time.time()-start, best_candidate, new_base_ot_approx))
            print('cost = ', cost + 1, 'time = ', time.time() - start, 'best = ', best_candidate, 'o_t_approx = ', new_base_ot_approx)
            base_ot_approx = new_base_ot_approx             # update the base o_t_approx for the next iteration
            ordered_insert(subset_index, best_candidate)    # guarantee subset_index always be sorted here
            subset_to_compute.append(copy.deepcopy(subset_index))
            plot_data.append([len(subset_index), base_ot_approx, 0]) # don't compute real o_t now, delay to after all the subsets are selected
            complement_sensors.remove(best_sensor)
            cost += 1
            if base_ot_approx > 0.9999999999999:
                break
        print('number of o_t_approx', counter)
        #return # for scalability test, we don't need to compute the real Ot in the scalability test.
        subset_results = Parallel(n_jobs=len(plot_data))(delayed(self.o_t_cpu)(subset_index) for subset_index in subset_to_compute)

        for i in range(len(subset_results)):
            plot_data[i][2] = subset_results[i]

        return plot_data

    def inner_greedy(self, subset_index, cuda_kernal, candidate):
        '''Inner loop for selecting candidates
        Args:
            subset_index (list):
            candidate (int):
        Return:
            (tuple): (index, o_t_approx, new subset_index)
        '''
        subset_index2 = copy.deepcopy(subset_index)
        ordered_insert(subset_index2, candidate)     # guarantee subset_index always be sorted here
        o_t = self.o_t_approx_host_old(subset_index2, cuda_kernal)
        return (candidate, o_t, subset_index2)


    #@profile
    def inner_greedy_real_ot_old(self, subset_index):
        '''Compute the real o_t (accruacy of prediction)
        Old function included only to compare performance improvement
        Args:
            subset_index (list):
        Return:
            (tuple): (index, o_t_approx, new subset_index)
        '''
        o_t = self.o_t_host(subset_index)
        return o_t

    def select_offline_greedy_lazy_gpu(self, budget, cores, cuda_kernal):
        '''(Parallel + Lazy greedy) Select a subset of sensors greedily. offline + homo version using ** GPU **
           The O(BS M^2) implementation + lookup table
        Args:
            budget (int): budget constraint
            cores (int): number of cores for parallelzation
            cuda_kernal (cuda_kernals.o_t_approx_kernal2 or o_t_approx_dist_kernal2): the O_{aux} in the paper
        Return:
            (list): an element is [str, int, float],
                    where str is the list of subset_index, int is # of sensors, float is O_T
        '''
        print('Start sensor selection...')
        start1 = time.time()
        base_ot_approx = 0
        if cuda_kernal == o_t_approx_kernal2:
            base_ot_approx = 1 - 0.5*len(self.transmitters)
        elif cuda_kernal == o_t_approx_dist_kernal2:
            largest_dist = (self.grid_len-1)*math.sqrt(2)
            max_gain_up_bound = 0.5*len(self.transmitters)*largest_dist   # the default bound is for non-distance
            for sensor in self.sensors:                                   # need to update the max gain upper bound for o_t_approx with distance
                sensor.gain_up_bound = max_gain_up_bound
            base_ot_approx = (1 - 0.5*len(self.transmitters))*largest_dist

        #self.means -= 5
        #self.meanvec_array -= 5
        self.rescale_intruder_hypothesis()

        plot_data = []
        cost = 0                                             # |T| in the paper
        subset_index = []                                    # T   in the paper
        complement_sensors = copy.deepcopy(self.sensors)     # S\T in the paper
        subset_to_compute = []
        n_h = len(self.transmitters)                         # number of hypotheses/transmitters
        dot_of_selected   = np.zeros((n_h, n_h))
        d_dot_of_selected = cuda.to_device(dot_of_selected)  # ValueError: ctypes objects containing pointers cannot be pickled
        d_covariance      = cuda.to_device(self.covariance)  # transfer only once
        d_meanvec         = cuda.to_device(self.meanvec_array)
        d_results         = cuda.device_array(n_h*n_h, np.float64)
        d_lookup_table    = cuda.to_device(self.lookup_table_q)


        #logger = open('dataSplat/log', 'w')
        while cost < budget and complement_sensors:
            best_candidate = complement_sensors[0].index    # init as the first sensor
            best_sensor = complement_sensors[0]
            complement_sensors.sort()                       # sorting the gain descendingly
            new_base_ot_approx = 0
            max_gain = 0
            #start = time.time()
            for i in range(len(complement_sensors)):
                candidate = complement_sensors[i].index

                candidate_result = self.o_t_approx_host(d_dot_of_selected, candidate,
                                                        d_covariance, d_meanvec, d_results, cuda_kernal, d_lookup_table)

                #print(i, (complement_sensors[i].x, complement_sensors[i].y), candidate_result)
                complement_sensors[i].gain_up_bound = candidate_result - base_ot_approx
                if complement_sensors[i].gain_up_bound > max_gain:
                    max_gain = complement_sensors[i].gain_up_bound
                    best_candidate = candidate
                    best_sensor = complement_sensors[i]
                    new_base_ot_approx = candidate_result

                if i+1 < len(complement_sensors) and max_gain > complement_sensors[i+1].gain_up_bound:   # where the lazy happens
                    #print('LAZY! ', cost, i, 'saves', len(complement_sensors) - i)
                    break


            self.update_dot_of_selected_host(d_dot_of_selected, best_candidate, d_covariance, d_meanvec)

            #print('cost = {}, time = {}, best = {}, ({}, {}), o_t = {}'.format(\
            #    cost+1, time.time()-start, best_candidate, best_sensor.x, best_sensor.y, new_base_ot_approx))
            #print(best_sensor.x, best_sensor.y, file=logger)
            base_ot_approx = new_base_ot_approx             # update the base o_t_approx for the next iteration
            ordered_insert(subset_index, best_candidate)    # guarantee subset_index always be sorted here
            subset_to_compute.append(copy.copy(subset_index))
            plot_data.append([len(subset_index), base_ot_approx, 0, copy.copy(subset_index)]) # don't compute real o_t now, delay to after all the subsets are selected
            complement_sensors.remove(best_sensor)
            if base_ot_approx > 0.9999999999999:
                break
            cost += 1
        #return # test speed for pure selection
        #logger.close()
        #print('Total time of selection: {:.3f} s'.format(time.time() - start1))
        start = time.time()
        subset_results = Parallel(n_jobs=cores, max_nbytes=None)(delayed(self.o_t_host)(subset_index) for subset_index in subset_to_compute)
        #print('Total time of optimal  : {:.3f} s'.format(time.time() - start))
        # for i in range(len(subset_index)):
        #     print(plot_data[i][1])
        for i in range(len(subset_results)):
            plot_data[i][2] = subset_results[i]
        return plot_data

    def select_offline_greedy_p_lazy_cpu_old(self, budget, cores):
        '''(Parallel + Lazy greedy) Select a subset of sensors greedily. offline + homo version using ** CPU **
        Attributes:
            budget (int): budget constraint
            cores (int): number of cores for parallelzation
        Return:
            (list): an element is [str, int, float],
                    where str is the list of subset_index, int is # of sensors, float is O_T
        '''
        counter = 0
        base_ot_approx = 1 - 0.5 * len(self.transmitters)
        plot_data = []
        cost = 0  # |T| in the paper
        subset_index = []  # T   in the paper
        complement_sensors = copy.deepcopy(self.sensors)  # S\T in the paper
        subset_to_compute = []
        while cost < budget and complement_sensors:
            best_candidate = -1
            best_sensor = None
            complement_sensors.sort()  # sorting the gain descendingly
            new_base_ot_approx = 0
            # for sensor in complement_sensors:
            #    print((sensor.index, sensor.gain_up_bound), end=' ')
            update, max_gain = 0, 0
            while update < len(complement_sensors):
                update_end = update + cores if update + cores <= len(complement_sensors) else len(
                    complement_sensors)
                candidiate_index = []
                for i in range(update, update_end):
                    candidiate_index.append(complement_sensors[i].index)
                counter += 1
                candidate_results = Parallel(n_jobs=cores)(
                    delayed(self.inner_greedy_cpu)(subset_index, candidate) for candidate in candidiate_index)
                # an element of candidate_results is a tuple - (index, o_t_approx, subsetlist)
                for i, j in zip(range(update, update_end), range(0,
                                                                 cores)):  # the two range might be different, if the case, follow the first range
                    complement_sensors[i].gain_up_bound = candidate_results[j][
                                                              1] - base_ot_approx  # update the upper bound of gain
                    if complement_sensors[i].gain_up_bound > max_gain:
                        max_gain = complement_sensors[i].gain_up_bound
                        best_candidate = candidate_results[j][0]
                        best_sensor = complement_sensors[i]
                        new_base_ot_approx = candidate_results[j][1]

                if update_end < len(complement_sensors) and max_gain > complement_sensors[
                    update_end].gain_up_bound:  # where the lazy happens
                    print('\n***LAZY!***\n', cost, (update, update_end), len(complement_sensors), '\n')
                    break
                update += cores
            base_ot_approx = new_base_ot_approx  # update the base o_t_approx for the next iteration
            print(best_candidate, base_ot_approx, '\n\n')
            ordered_insert(subset_index, best_candidate)  # guarantee subset_index always be sorted here
            subset_to_compute.append(copy.deepcopy(subset_index))
            plot_data.append([len(subset_index), base_ot_approx,
                              0])  # don't compute real o_t now, delay to after all the subsets are selected
            complement_sensors.remove(best_sensor)
            cost += 1
            if base_ot_approx > 0.9999999999999:
                break
        print('number of o_t_approx', counter)
        #return  # for scalability test, we don't need to compute the real Ot in the scalability test.
        subset_results = Parallel(n_jobs=len(plot_data))(
            delayed(self.inner_greedy_real_ot_cpu)(subset_index) for subset_index in subset_to_compute)

        for i in range(len(subset_results)):
            plot_data[i][2] = subset_results[i]

        return plot_data

    def inner_greedy_cpu(self, subset_index, candidate):
        '''Inner loop for selecting candidates
        Args:
            subset_index (list):
            candidate (int):
        Return:
            (tuple): (index, o_t_approx, new subset_index)
        '''
        subset_index2 = copy.deepcopy(subset_index)
        ordered_insert(subset_index2, candidate)  # guarantee subset_index always be sorted here
        o_t = self.o_t_approximate(subset_index2)
        return (candidate, o_t, subset_index2)

    # @profile
    def select_offline_GA_old(self, budget, cores):
        '''Using the Ot real during selection, not submodular, no proformance guarantee
        Args:
            budget (int): budget constraint
            cores (int): number of cores for parallelzation
        Return:
            (list): an element is [str, int, float],
                    where str is the list of subset_index, int is # of sensors, float is O_T
        '''
        print('Start GA selection (homo)')
        plot_data = []
        cost = 0                                            # |T| in the paper
        subset_index = np.array([])                         # T   in the paper
        complement_index = np.array([i for i in range(self.sen_num)]).astype(int) # S\T in the paper
        while cost < budget and len(complement_index) > 0:
            start = time.time()
            candidate_results = [self.inner_greedy_real(subset_index, candidate) for candidate in complement_index]
            all_candidate_ot = [cr[1] for cr in candidate_results]
            best_candidate = np.argmax(all_candidate_ot)
            best_sensor = complement_index[best_candidate]
            maximum = all_candidate_ot[best_candidate]
            print('cost = {}, time = {}, best = {}, ({}, {}), o_t = {}'.format(\
                cost+1, time.time()-start, best_sensor, self.sensors[best_sensor].x, self.sensors[best_sensor].y, maximum))

            subset_index = np.append(subset_index, best_sensor)
            subset_index = np.partition(subset_index, len(subset_index) - 1).astype(int)
            complement_index = np.delete(complement_index, best_candidate)
            cost += 1
            plot_data.append([len(subset_index), maximum, subset_index])

            if maximum > 0.999999999:
                break

        return plot_data


    # @profile
    def select_offline_GA(self, budget, cuda_kernal):
        '''Using the Ot real during selection, not submodular, no proformance guarantee
        Args:
            budget (int): budget constraint
            cores (int): number of cores for parallelzation
        Return:
            (list): an element is [str, int, float],
                    where str is the list of subset_index, int is # of sensors, float is O_T
        '''
        print('Start GA selection (homo)')
        plot_data = []
        cost = 0                                              # |T| in the paper
        subset_index = []                                     # T   in the paper
        complement_index = [i for i in range(self.sen_num)]   # S\T in the paper
        n_h = len(self.transmitters)
        dot_of_selected   = np.zeros((n_h, n_h))
        d_dot_of_selected = cuda.to_device(dot_of_selected)
        d_covariance      = cuda.to_device(self.covariance)
        d_meanvec         = cuda.to_device(self.meanvec_array)
        d_results         = cuda.device_array((n_h, n_h), np.float64)
        d_lookup_table    = cuda.to_device(self.lookup_table_q)

        while cost < budget and len(complement_index) > 0:
            start = time.time()

            candidate_results = [self.o_t_host_iter(d_dot_of_selected, candidate, d_covariance, d_meanvec, d_results, cuda_kernal, d_lookup_table)
                                 for candidate in complement_index]
        
            best_candidate = np.argmax(candidate_results)
            best_sensor = complement_index[best_candidate]
            maximum = candidate_results[best_candidate]
            print('cost = {}, time = {}, best = {}, ({}, {}), o_t = {}'.format(\
                cost+1, time.time()-start, best_sensor, self.sensors[best_sensor].x, self.sensors[best_sensor].y, maximum))

            self.update_dot_of_selected_host(d_dot_of_selected, best_sensor, d_covariance, d_meanvec)

            subset_index = np.append(subset_index, best_sensor)
            subset_index = np.partition(subset_index, len(subset_index) - 1).astype(int)
            complement_index = np.delete(complement_index, best_candidate)
            cost += 1
            plot_data.append([len(subset_index), maximum, subset_index])

            if maximum > 0.999999999:
                break

        return plot_data

    def select_offline_GA_hetero(self, budget, cuda_kernal):
        print('Start GA selection (hetero)')
        cost = 0  # |T| in the paper
        subset_index = []  # T   in the paper
        complement_index = [i for i in range(self.sen_num)]  # S\T in the paper
        maximum = 0
        n_h = len(self.transmitters)
        dot_of_selected = np.zeros((n_h, n_h))
        d_dot_of_selected = cuda.to_device(dot_of_selected)
        d_covariance = cuda.to_device(self.covariance)
        d_meanvec = cuda.to_device(self.meanvec_array)
        d_results = cuda.device_array((n_h, n_h), np.float64)
        d_lookup_table = cuda.to_device(self.lookup_table_q)
        sensor_costs = np.array([self.sensors[i].cost for i in range(len(self.sensors))])

        first_pass_plot_data = []
        while cost < budget and len(complement_index) > 0:
            start = time.time()

            candidate_results = [
                self.o_t_host_iter(d_dot_of_selected, candidate, d_covariance, d_meanvec, d_results, cuda_kernal,
                                   d_lookup_table)
                for candidate in complement_index]

            best_candidate = np.argmax(candidate_results)
            best_sensor = complement_index[best_candidate]
            maximum = candidate_results[best_candidate]
            print('cost = {}, time = {}, best = {}, ({}, {}), o_t = {}'.format( \
                cost + 1, time.time() - start, best_sensor, self.sensors[best_sensor].x, self.sensors[best_sensor].y,
                maximum))

            self.update_dot_of_selected_host(d_dot_of_selected, best_sensor, d_covariance, d_meanvec)

            subset_index = np.append(subset_index, best_sensor)
            subset_index = np.partition(subset_index, len(subset_index) - 1).astype(int)
            complement_index = np.delete(complement_index, best_candidate)
            cost += sensor_costs[best_sensor]
            first_pass_plot_data.append([len(subset_index), maximum, subset_index])

            if maximum > 0.999999999:
                break
        print('end of the first homo pass and start of the second hetero pass')

        cost = 0  # |T| in the paper
        subset_index = []  # T   in the paper
        complement_index = [i for i in range(self.sen_num)]  # S\T in the paper
        base_ot = 0  # O_T from the previous iteration
        dot_of_selected = np.zeros((n_h, n_h))
        d_dot_of_selected = cuda.to_device(dot_of_selected)
        d_covariance = cuda.to_device(self.covariance)
        d_meanvec = cuda.to_device(self.meanvec_array)
        d_results = cuda.device_array((n_h, n_h), np.float64)
        d_lookup_table = cuda.to_device(self.lookup_table_q)
        second_pass_plot_data = []
        while cost < budget and len(complement_index) > 0:
            start = time.time()

            candidate_results = [
                self.o_t_host_iter(d_dot_of_selected, candidate, d_covariance, d_meanvec, d_results, cuda_kernal,
                                   d_lookup_table)
                for candidate in complement_index]

            result_to_cost_ratio = np.divide(candidate_results, sensor_costs)
            best_candidate = np.argmax(result_to_cost_ratio)
            best_sensor = complement_index[best_candidate]
            maximum = candidate_results[best_candidate]
            print('cost = {}, time = {}, best = {}, ({}, {}), o_t = {}'.format( \
                cost + 1, time.time() - start, best_sensor, self.sensors[best_sensor].x, self.sensors[best_sensor].y,
                maximum))

            self.update_dot_of_selected_host(d_dot_of_selected, best_sensor, d_covariance, d_meanvec)

            subset_index = np.append(subset_index, best_sensor)
            subset_index = np.partition(subset_index, len(subset_index) - 1).astype(int)
            complement_index = np.delete(complement_index, best_candidate)
            cost += sensor_costs[best_candidate]
            sensor_costs = np.delete(sensor_costs, best_candidate)

            second_pass_plot_data.append([len(subset_index), maximum, subset_index])

            if maximum > 0.999999999:
                break
        print('end of the first homo pass and start of the second hetero pass')

        final_result = []
        if second_pass_plot_data[-1][1] > first_pass_plot_data[-1][1]:
            return second_pass_plot_data
        else:
            return first_pass_plot_data

    def select_offline_GA_hetero_old(self, budget, cores):
        '''Offline selection when the sensors are heterogeneous
           Two pass method: first do a homo pass, then do a hetero pass, choose the best of the two
        Args:
            budget (int): budget we have for the heterogeneous sensors
            cores (int): number of cores for parallelization
        '''
        print('Start GA selection (hetero)')
        cost = 0                                             # |T| in the paper
        subset_index = []                                    # T   in the paper
        complement_index = [i for i in range(self.sen_num)]  # S\T in the paper
        maximum = 0
        first_pass_plot_data = []
        while cost < budget and complement_index:
            sensor_delete = []
            for index in complement_index:
                if cost + self.sensors[index].cost > budget: # over budget
                    sensor_delete.append(index)
            for sensor in sensor_delete:
                complement_index.remove(sensor)
            if not complement_index:                         # if there are no sensors that can be selected, then break
                break

            candidate_results = Parallel(n_jobs=cores, max_nbytes=None)(delayed(self.inner_greedy_real)(subset_index, candidate) for candidate in complement_index)

            best_candidate = candidate_results[0][0]   # an element of candidate_results is a tuple - (int, float, list)
            maximum = candidate_results[0][1]          # where int is the candidate, float is the O_T, list is the subset_list with new candidate
            for candidate in candidate_results:
                if candidate[1] > maximum:
                    best_candidate = candidate[0]
                    maximum = candidate[1]

            ordered_insert(subset_index, best_candidate)    # guarantee subset_index always be sorted here
            complement_index.remove(best_candidate)
            cost += self.sensors[best_candidate].cost
            first_pass_plot_data.append([cost, maximum, copy.copy(subset_index)])           # Y value is real o_t
            print(best_candidate, maximum, cost)

        print('end of the first homo pass and start of the second hetero pass')

        cost = 0                                            # |T| in the paper
        subset_index = []                                   # T   in the paper
        complement_index = [i for i in range(self.sen_num)] # S\T in the paper
        base_ot = 0                                         # O_T from the previous iteration
        second_pass_plot_data = []
        while cost < budget and complement_index:
            sensor_delete = []
            for index in complement_index:
                if cost + self.sensors[index].cost > budget: # over budget
                    sensor_delete.append(index)
            for sensor in sensor_delete:
                complement_index.remove(sensor)
            if not complement_index:                         # if there are no sensors that can be selected, then break
                break

            candidate_results = Parallel(n_jobs=cores, max_nbytes=None)(delayed(self.inner_greedy_real)(subset_index, candidate) for candidate in complement_index)

            best_candidate = candidate_results[0][0]                       # an element of candidate_results is a tuple - (int, float, list)
            cost_of_candiate = self.sensors[best_candidate].cost
            new_base_ot = candidate_results[0][1]
            maximum = (candidate_results[0][1]-base_ot)/cost_of_candiate   # where int is the candidate, float is the O_T, list is the subset_list with new candidate
            for candidate in candidate_results:
                incre = candidate[1] - base_ot
                cost_of_candiate = self.sensors[candidate[0]].cost
                incre_cost = incre/cost_of_candiate     # increment of O_T devided by cost
                #print(candidate[2], candidate[1], incre, cost_of_candiate, incre_cost)
                if incre_cost > maximum:
                    best_candidate = candidate[0]
                    maximum = incre_cost
                    new_base_ot = candidate[1]
            base_ot = new_base_ot
            ordered_insert(subset_index, best_candidate)    # guarantee subset_index always be sorted here
            complement_index.remove(best_candidate)
            cost += self.sensors[best_candidate].cost
            second_pass_plot_data.append([cost, base_ot, copy.copy(subset_index)])           # Y value is real o_t
            print(best_candidate, base_ot, cost)

        if second_pass_plot_data[-1][1] > first_pass_plot_data[-1][1]:
            print('second pass is selected')
            return second_pass_plot_data
        else:
            print('first pass is selected')
            return first_pass_plot_data

    def select_offline_optimal(self, budget, cores, num_samples = 100):
        '''brute force all possible subsets in a small input such as 10 x 10 grid
        Args:
            budget (int): budget constraint
            cores  (int): number of cores for parallelzation
        Return:
            (list): an element is [int, float, list],
                    where str is int is # of sensors, float is O_T, list of subset_index
        '''
        # start = time.time()
        # subset_to_compute = list(combinations(range(len(self.sensors)), budget))
        # try:
        #     subset_to_compute = random.sample(subset_to_compute, num_samples)
        # except:
        #     pass
        start = time.time()
        subset_to_compute = [0] * num_samples
        for i in range(num_samples):
            subset_to_compute[i] = np.random.choice(len(self.sensors), budget, replace=False)

        print('cost = {}, # Ot = {},'.format(budget, len(subset_to_compute)), end=' ')
        results = [self.o_t_host(subset_index) for subset_index in subset_to_compute]

        results = np.array(results)
        best_subset = results.argmax()
        best_ot = results.max()
        print('time = {}, best subset = {}, best Ot = {}'.format( \
               time.time()-start, subset_to_compute[best_subset], best_ot))

        return budget, best_ot


    def inner_greedy_real(self, subset_index, candidate):
        '''Inner loop for selecting candidates of GA
        Args:
            subset_index (list):
            candidate (int):
        Return:
            (tuple): (index, o_t_approx, new subset_index)
        '''
        subset_index2 = np.append(subset_index, candidate).astype(int)
        subset_index2 = np.partition(subset_index2, len(subset_index2) - 1)
        #print(subset_index2)
        #subset_index2 = subset_index.copy()
        #subset_index2 = ordered_insert(subset_index2, candidate)     # guarantee subset_index always be sorted here
        #actual_time_start = time.time()
        o_t = self.o_t_host(subset_index2)
        #actual_finish_time = time.time()
        return (candidate, o_t, subset_index2)


    def select_offline_random(self, number, cores):
        '''Select a subset of sensors randomly
        Args:
            number (int): number of sensors to be randomly selected
            cores (int): number of cores for parallelization
        Return:
            (list): results to be plotted. each element is (str, int, float),
                    where str is the list of selected sensors, int is # of sensor, float is O_T
        '''
        print('Start random sensor selection (homo)')
        random.seed()
        subset_index = []
        plot_data = []
        sequence = [i for i in range(self.sen_num)]
        i = 1

        subset_to_compute = []
        while i <= number:
            select = random.choice(sequence)
            ordered_insert(subset_index, select)
            subset_to_compute.append(copy.deepcopy(subset_index))
            sequence.remove(select)
            i += 1

        results = Parallel(n_jobs=cores, max_nbytes=None)(delayed(self.o_t_host)(subset_index) for subset_index in subset_to_compute)

        for subset, result in zip(subset_to_compute, results):
            plot_data.append([len(subset), result, subset])

        return plot_data


    def select_offline_random_hetero(self, budget, cores):
        '''Offline selection when the sensors are heterogeneous
        Args:
            budget (int): budget we have for the heterogeneous sensors
            cores (int): number of cores for parallelization
        '''
        print('Start random sensor selection (hetero)')
        random.seed(0)    # though algorithm is random, the results are the same every time

        self.subset = {}
        subset_index = []
        sequence = [i for i in range(self.sen_num)]
        cost = 0
        cost_list = []
        subset_to_compute = []
        while cost < budget:
            option = []
            for index in sequence:
                temp_cost = self.sensors[index].cost
                if cost + temp_cost <= budget:  # a sensor can be selected if adding its cost is under budget
                    option.append(index)
            if not option:                      # if there are no sensors that can be selected, then break
                break
            select = random.choice(option)
            ordered_insert(subset_index, select)
            subset_to_compute.append(copy.deepcopy(subset_index))
            sequence.remove(select)
            cost += self.sensors[select].cost
            cost_list.append(cost)

        results = Parallel(n_jobs=cores, max_nbytes=None)(delayed(self.o_t_host)(subset_index) for subset_index in subset_to_compute)

        plot_data = []
        for cost, result, subset in zip(cost_list, results, subset_to_compute):
            plot_data.append([cost, result, subset])

        return plot_data


    def select_offline_coverage(self, budget, cores):
        '''A coverage-based baseline algorithm
        '''
        print('start coverage-based selection (homo)')
        random.seed(0)
        center = (int(self.grid_len/2), int(self.grid_len/2))
        min_dis = 99999
        first_index, i = 0, 0
        first_sensor = None
        for sensor in self.sensors:        # select the first sensor that is closest to the center of the grid
            temp_dis = distance.euclidean([center[0], center[1]], [sensor.x, sensor.y])
            if temp_dis < min_dis:
                min_dis = temp_dis
                first_index = i
                first_sensor = sensor
            i += 1
        subset_index = [first_index]
        subset_to_compute = [copy.deepcopy(subset_index)]
        complement_index = [i for i in range(self.sen_num)]
        complement_index.remove(first_index)

        radius = self.compute_coverage_radius(first_sensor, subset_index) # compute the radius
        print('radius', radius)
        coverage = np.zeros((self.grid_len, self.grid_len), dtype=int)
        self.add_coverage(coverage, first_sensor, radius)
        cost = 1
        while cost < budget and complement_index:  # find the sensor that has the least overlap
            least_overlap = 99999
            best_candidate = []
            best_sensor = []
            for candidate in complement_index:
                sensor = self.index_to_sensor(candidate)
                overlap = self.compute_overlap(coverage, sensor, radius)
                if overlap < least_overlap:
                    least_overlap = overlap
                    best_candidate = [candidate]
                    best_sensor = [sensor]
                elif overlap == least_overlap:
                    best_candidate.append(candidate)
                    best_sensor.append(sensor)
            choose = random.choice(range(len(best_candidate)))
            ordered_insert(subset_index, best_candidate[choose])
            complement_index.remove(best_candidate[choose])
            self.add_coverage(coverage, best_sensor[choose], radius)
            subset_to_compute.append(copy.deepcopy(subset_index))
            cost += 1

        results = Parallel(n_jobs=cores, max_nbytes=None)(delayed(self.o_t_host)(subset_index) for subset_index in subset_to_compute)

        plot_data = []
        for subset, result in zip(subset_to_compute, results):
            plot_data.append([len(subset), result, subset])

        return plot_data


    def select_offline_coverage_hetero(self, budget, cores):
        '''A coverage-based baseline algorithm (heterogeneous version)
        '''
        print('Start coverage-based sensor selection (hetero)')

        random.seed(0)

        center = (int(self.grid_len/2), int(self.grid_len/2))
        min_dis = 99999
        first_index, i = 0, 0
        first_sensor = None
        for sensor in self.sensors:        # select the first sensor that is closest to the center of the grid
            temp_dis = distance.euclidean([center[0], center[1]], [sensor.x, sensor.y])
            if temp_dis < min_dis:
                min_dis = temp_dis
                first_index = i
                first_sensor = sensor
            i += 1
        subset_index = [first_index]
        subset_to_compute = [copy.deepcopy(subset_index)]
        complement_index = [i for i in range(self.sen_num)]
        complement_index.remove(first_index)

        radius = self.compute_coverage_radius(first_sensor, subset_index) # compute the radius
        print('radius', radius)

        coverage = np.zeros((self.grid_len, self.grid_len), dtype=int)
        self.add_coverage(coverage, first_sensor, radius)
        cost = self.sensors[first_index].cost
        cost_list = [cost]

        while cost < budget and complement_index:
            option = []
            for index in complement_index:
                temp_cost = self.sensors[index].cost
                if cost + temp_cost <= budget:  # a sensor can be selected if adding its cost is under budget
                    option.append(index)
            if not option:                      # if there are no sensors that can be selected, then break
                break

            min_overlap_cost = 99999   # to minimize overlap*cost
            best_candidate = []
            best_sensor = []
            for candidate in option:
                sensor = self.index_to_sensor(candidate)
                overlap = self.compute_overlap(coverage, sensor, radius)
                temp_cost = self.sensors[candidate].cost
                overlap_cost = (overlap+0.001)*temp_cost
                if overlap_cost < min_overlap_cost:
                    min_overlap_cost = overlap_cost
                    best_candidate = [candidate]
                    best_sensor = [sensor]
                elif overlap_cost == min_overlap_cost:
                    best_candidate.append(candidate)
                    best_sensor.append(sensor)
            choose = random.choice(range(len(best_candidate)))
            ordered_insert(subset_index, best_candidate[choose])
            complement_index.remove(best_candidate[choose])
            self.add_coverage(coverage, best_sensor[choose], radius)
            subset_to_compute.append(copy.deepcopy(subset_index))
            cost += self.sensors[best_candidate[choose]].cost
            cost_list.append(cost)

        print(len(subset_to_compute), subset_to_compute)
        results = Parallel(n_jobs=cores, max_nbytes=None)(delayed(self.o_t_host)(subset_index) for subset_index in subset_to_compute)

        plot_data = []
        for cost, result, subset in zip(cost_list, results, subset_to_compute):
            plot_data.append([cost, result, subset])

        return plot_data


    def select_offline_greedy_hetero(self, budget, cores, cuda_kernal):
        '''(Lazy) Offline selection when the sensors are heterogeneous
           Two pass method: first do a homo pass, then do a hetero pass, choose the best of the two
        Args:
            budget (int): budget we have for the heterogeneous sensors
            cores (int): number of cores for parallelization
            cost_filename (str): file that has the cost of sensors
        '''
        print('Start sensor selection (hetero)')
        base_ot_approx = 1 - 0.5*len(self.transmitters)
        cost = 0                                            # |T| in the paper
        subset_index = []                                   # T   in the paper
        complement_sensors = copy.deepcopy(self.sensors)    # S\T in the paper
        first_pass_plot_data = []

        n_h = len(self.transmitters)
        dot_of_selected   = np.zeros((n_h, n_h))
        d_dot_of_selected = cuda.to_device(dot_of_selected)
        d_covariance      = cuda.to_device(self.covariance)
        d_meanvec         = cuda.to_device(self.meanvec_array)
        d_results         = cuda.device_array(n_h*n_h, np.float64)
        d_lookup_table    = cuda.to_device(self.lookup_table_q)

        while cost < budget and complement_sensors:
            sensor_delete = []                  # sensors that will lead to over budget
            for sensor in complement_sensors:
                if cost + sensor.cost > budget: # over budget
                    sensor_delete.append(sensor)
            for sensor in sensor_delete:
                complement_sensors.remove(sensor)
            complement_sensors.sort()           # sort the sensors by gain upper bound descendingly
            if not complement_sensors:          # if there are no sensors that can be selected, then break
                break

            best_candidate = complement_sensors[0].index
            best_sensor = complement_sensors[0]
            new_base_ot_approx = 0
            max_gain = 0

            for i in range(len(complement_sensors)):
                candidate = complement_sensors[i].index
                candidate_result = self.o_t_approx_host(d_dot_of_selected, candidate, d_covariance, d_meanvec, d_results, cuda_kernal, d_lookup_table)

                complement_sensors[i].gain_up_bound = candidate_result - base_ot_approx
                if complement_sensors[i].gain_up_bound > max_gain:
                    max_gain = complement_sensors[i].gain_up_bound
                    best_candidate = candidate
                    best_sensor = complement_sensors[i]
                    new_base_ot_approx = candidate_result

                if i+1 < len(complement_sensors) and max_gain > complement_sensors[i+1].gain_up_bound:   # where the lazy happens
                    #print('\n******LAZY! cost, (update, update_end), len(complement_sensors)')
                    break

            self.update_dot_of_selected_host(d_dot_of_selected, best_candidate, d_covariance, d_meanvec)

            base_ot_approx = new_base_ot_approx
            ordered_insert(subset_index, best_candidate)    # guarantee subset_index always be sorted here
            complement_sensors.remove(best_sensor)
            cost += self.sensors[best_candidate].cost
            first_pass_plot_data.append([cost, 0, copy.copy(subset_index)])           # Y value is real o_t
            #print(best_candidate, base_ot_approx, cost)

        #print('Homo pass ends, hetero pass starts', end=' ')

        lowest_cost = 1
        for sensor in self.sensors:
            if sensor.cost < lowest_cost:
                lowest_cost = sensor.cost
        max_gain_up_bound = 0.5*len(self.transmitters)/lowest_cost
        for sensor in self.sensors:
            sensor.gain_up_bound = max_gain_up_bound

        cost = 0                                            # |T| in the paper
        subset_index = []                                   # T   in the paper
        complement_sensors = copy.copy(self.sensors)        # S\T in the paper
        base_ot_approx = 1 - 0.5*len(self.transmitters)
        second_pass_plot_data = []
        dot_of_selected   = np.zeros((n_h, n_h))
        d_dot_of_selected = cuda.to_device(dot_of_selected)

        while cost < budget and complement_sensors:
            sensor_delete = []                  # sensors that will lead to over budget
            for sensor in complement_sensors:
                if cost + sensor.cost > budget: # over budget
                    sensor_delete.append(sensor)
            for sensor in sensor_delete:
                complement_sensors.remove(sensor)
            complement_sensors.sort()           # sort the sensors by gain upper bound descendingly
            if not complement_sensors:          # if there are no sensors that can be selected, then break
                break

            best_candidate = complement_sensors[0].index
            best_sensor = complement_sensors[0]
            new_base_ot_approx = 0
            max_gain = 0

            for i in range(len(complement_sensors)):
                candidate = complement_sensors[i].index
                candidate_result = self.o_t_approx_host(d_dot_of_selected, candidate, d_covariance, d_meanvec, d_results, cuda_kernal, d_lookup_table)

                complement_sensors[i].gain_up_bound = (candidate_result - base_ot_approx)/complement_sensors[i].cost  # takes cost into account
                if complement_sensors[i].gain_up_bound > max_gain:
                    max_gain = complement_sensors[i].gain_up_bound
                    best_candidate = candidate
                    best_sensor = complement_sensors[i]
                    new_base_ot_approx = candidate_result

                if i+1 < len(complement_sensors) and max_gain > complement_sensors[i+1].gain_up_bound:   # where the lazy happens
                    #print('\n******LAZY! cost, (update, update_end), len(complement_sensors)')
                    break

            self.update_dot_of_selected_host(d_dot_of_selected, best_candidate, d_covariance, d_meanvec)
            base_ot_approx = new_base_ot_approx
            ordered_insert(subset_index, best_candidate)    # guarantee subset_index always be sorted here
            complement_sensors.remove(best_sensor)
            cost += self.sensors[best_candidate].cost
            second_pass_plot_data.append([cost, 0, copy.copy(subset_index)])           # Y value is real o_t
            #print(best_candidate, base_ot_approx, cost)

        first_pass = []
        for data in first_pass_plot_data:
            first_pass.append(data[2])
        second_pass = []
        for data in second_pass_plot_data:
            second_pass.append(data[2])

        first_pass_o_ts = Parallel(n_jobs=cores, max_nbytes=None)(delayed(self.o_t_host)(subset_index) for subset_index in first_pass)
        second_pass_o_ts = Parallel(n_jobs=cores, max_nbytes=None)(delayed(self.o_t_host)(subset_index) for subset_index in second_pass)

        print(first_pass_o_ts)

        #first_cost_pos = 0
        #second_cost_pos = 0
        if second_pass_o_ts[-1] > first_pass_o_ts[-1]:
            print('Second pass is selected')
            for i in range(len(second_pass_o_ts)):
                second_pass_plot_data[i][1] = second_pass_o_ts[i]
            return second_pass_plot_data
        else:
            print('First pass is selected')
            for i in range(len(first_pass_o_ts)):
                first_pass_plot_data[i][1] = first_pass_o_ts[i]
            return first_pass_plot_data


    def update_battery(self, selected, energy=1):
        '''Update the battery of sensors
        Args:
            energy (int):    energy consumption amount
            selected (list): list of index of selected sensors
        '''
        for select in selected:
            self.sensors[select].update_battery(energy)


    def index_to_sensor(self, index):
        '''A temporary solution for the inappropriate data structure for self.sensors
        '''
        i = 0
        for sensor in self.sensors:
            if i == index:
                return sensor
            else:
                i += 1


    def compute_coverage_radius(self, first_sensor, subset_index):
        '''Compute the coverage radius for the coverage-based selection algorithm
        Args:
            first_sensor (tuple): sensor that is closest to the center
            subset_index (list):
        '''
        return 3
        sub_cov = self.covariance_sub(subset_index)
        sub_cov_inv = np.linalg.inv(sub_cov)        # inverse
        radius = 1
        for i in range(1, int(self.grid_len/2)):    # compute 'radius'
            transmitter_i = self.transmitters[(first_sensor.x - i)*self.grid_len + first_sensor.y] # 2D index --> 1D index
            i_x, i_y = transmitter_i.x, transmitter_i.y
            if i_x < 0:
                break
            transmitter_i.set_mean_vec_sub(subset_index)
            prob_i = []
            for transmitter_j in self.transmitters:
                j_x, j_y = transmitter_j.x, transmitter_j.y
                if i_x == j_x and i_y == j_y:
                    continue
                transmitter_j.set_mean_vec_sub(subset_index)
                pj_pi = transmitter_j.mean_vec_sub - transmitter_i.mean_vec_sub
                prob_i.append(1 - norm.sf(0.5 * math.sqrt(np.dot(np.dot(pj_pi, sub_cov_inv), pj_pi))))
            product = 1
            for prob in prob_i:
                product *= prob
            print(i, product)
            if product > 0.00001:     # set threshold
                radius = i
            else:
                break
        return radius


    def compute_overlap(self, coverage, sensor, radius):
        '''Compute the overlap between selected sensors and the new sensor
        Args:
            coverage (2D array)
            sensor (Sensor)
            radius (int)
        '''
        x_low = sensor.x - radius if sensor.x - radius >= 0 else 0
        x_high = sensor.x + radius if sensor.x + radius <= self.grid_len-1 else self.grid_len-1
        y_low = sensor.y - radius if sensor.y - radius >= 0 else 0
        y_high = sensor.y + radius if sensor.y + radius <= self.grid_len-1 else self.grid_len-1

        overlap = 0
        for x in range(x_low, x_high+1):
            for y in range(y_low, y_high):
                if distance.euclidean([x, y], [sensor.x, sensor.y]) <= radius:
                    overlap += coverage[x][y]
        return overlap


    def add_coverage(self, coverage, sensor, radius):
        '''When seleted a sensor, add coverage by 1
        Args:
            coverage (2D array): each element is a counter for coverage
            sensor (Sensor): (x, y)
            radius (int): radius of a sensor
        '''
        x_low = sensor.x - radius if sensor.x - radius >= 0 else 0
        x_high = sensor.x + radius if sensor.x + radius <= self.grid_len-1 else self.grid_len-1
        y_low = sensor.y - radius if sensor.y - radius >= 0 else 0
        y_high = sensor.y + radius if sensor.y + radius <= self.grid_len-1 else self.grid_len-1

        for x in range(x_low, x_high+1):
            for y in range(y_low, y_high+1):
                if distance.euclidean([x, y], [sensor.x, sensor.y]) <= radius:
                    coverage[x][y] += 1


   def compute_conditional_error(self, true_x, true_y, subset_index):
        '''Use Bayes formula to update P(hypothesis): from prior to posterior
           After we add a new sensor and get a larger subset, the larger subset begins to observe data from true transmitter
           An important update from update_hypothesis to update_hypothesis_2 is that we are not using attribute transmitter.multivariant_g
aussian. It saves money
        Args:
            true_transmitter (Transmitter)
            subset_index (list)
        '''
        # np.random.seed(true_x*self.grid_len + true_y*true_y)  # change seed here
        # data = np.zeros(len(subset_index))  # the true transmitter generate some data
        # for i, index in enumerate(subset_index):
        #     sensor = self.sensors[index]
        mean = self.means_rescale[self.grid_len * true_x + true_y, subset_index]
        std = self.stds[self.grid_len * true_x + true_y, subset_index]
        data = np.random.normal(mean, std)
        cov_sub = np.diagonal(self.covariance[np.ix_(subset_index, subset_index)])

        #distance = np.zeros(self.grid_len, self.grid_len)
        mean_vec_sub = np.array([trans.mean_vec[subset_index] for trans in self.transmitters])
        cov_sub = np.broadcast_to(cov_sub,(len(self.transmitters),len(cov_sub)))
        data = np.broadcast_to(data, (len(self.transmitters), len(data)))

        array_of_pdfs = norm(mean_vec_sub, cov_sub).pdf(data)
        likelihood = np.prod(array_of_pdfs, axis=1) #One likelihood for each transmitter
        #print(mean_vec_sub.shape, array_of_pdfs.shape, likelihood.shape)
        self.grid_posterior = np.multiply(likelihood, self.grid_priori.flatten())
        self.grid_posterior = np.multiply(self.grid_posterior, self.present)
        #print('denominator = ', denominator)
        denominator = np.sum(self.grid_posterior)
        if denominator == 0:
            self.grid_posterior = np.full((self.grid_len, self.grid_len), 1.0 / (self.grid_len * self.grid_len))
            denominator = np.sum(self.grid_posterior)
        self.grid_posterior /= denominator
        #self.grid_posterior = np.reshape(self.grid_posterior, (-1, self.grid_len))
        # for trans in self.transmitters:
        #     trans.set_mean_vec_sub(subset_index)
        #     array_of_pdfs = norm(trans.mean_vec_sub, cov_sub).pdf(data)
        #     likelihood = np.prod(array_of_pdfs)
        #     #print('Likelihood = ', likelihood)
        #     self.grid_posterior[trans.x][trans.y] = likelihood * self.grid_priori[trans.x][trans.y]
        # denominator = np.sum(self.grid_posterior)
        # #print('den = ', denominator)
        # self.grid_posterior = self.grid_posterior / denominator
        x_dist = np.array([trans.x - true_x for trans in self.transmitters])
        y_dist = np.array([trans.y - true_y for trans in self.transmitters])
        distance = np.sqrt(np.multiply(x_dist, x_dist) + np.multiply(y_dist, y_dist))
        #distance = np.reshape(distance, (-1, self.grid_len))
        #print("Max values=", np.max(self.grid_posterior), np.max(distance))
        self.grid_posterior = np.nan_to_num(self.grid_posterior)
        error = np.sum(np.multiply(distance, self.grid_posterior.flatten()))
        #np.set_printoptions(threshold=np.infty)
        #print(self.grid_posterior)

        # for trans in self.transmitters:
        #     distance = np.sqrt((true_x - trans.x) ** 2 + (true_y - trans.y) ** 2)
        #     error += self.grid_posterior[trans.x][trans.y] * distance
        return error




    def weighted_distance_priori(self, complement_index):
        '''Compute the weighted distance priori according to the priori distribution for every sensor in
           the complement index list and return the all the distances
        Args:
            complement_index (list)
        Return:
            (np.ndarray) - index
        '''
        distances = []
        for index in complement_index:
            sensor = self.sensors[index]
            weighted_distance = 0
            for transmitter in self.transmitters:
                tran_x, tran_y = transmitter.x, transmitter.y
                dist = distance.euclidean([sensor.x, sensor.y], [tran_x, tran_y])
                dist = dist if dist >= 1 else 0.5                                 # sensor very close by with high priori should be selected
                weighted_distance += 1/dist * self.grid_priori[tran_x][tran_y]    # so the metric is priori/disctance

            distances.append(weighted_distance)
        return np.array(distances)


    def transmitters_to_array(self):
        '''transform the transmitter objects to numpy array, for the sake of CUDA
        '''
        mylist = []
        for transmitter in self.transmitters:
            templist = []
            for mean in transmitter.mean_vec:
                templist.append(mean)
            mylist.append(templist)
        self.meanvec_array = np.array(mylist)  # TODO replace this with sels.means_all?


    #@profile
    def o_t_approx_host(self, d_dot_of_selected, candidate, d_covariance, d_meanvec, d_results, cuda_kernal, d_lookup_table):
        '''host code for o_t_approx.
            TYPE = "numba.cuda.cudadrv.devicearray.DeviceNDArray", which cannot be pickled --> cannot exist before using joblib
        Args:
            d_dot_of_selected (TYPE): stores the np.dot(np.dot(pj_pi, sub_cov_inv), pj_pi)) of sensors already selected
                                      in previous iterations. shape=(m, m) where m is the number of hypothesis (grid_len^2)
            candidate (int)         : a candidate sensor index
            d_covariance (TYPE)     : covariance matrix
            d_meanvec (TYPE)        : contains the mean vector of every transmitter
            d_results (TYPE)        : 1D array. save the results for each (i, j) pair of transmitter and sensor's error
            cuda_kernal (cuda_kernals.o_t_approx_kernal2 or o_t_approx_dist_kernal2)
            d_lookup_table (TYPE)   : trade space for time
        Return:
            (float): o_t_approx
        '''
        n_h = len(self.transmitters)
        threadsperblock = (self.TPB, self.TPB)
        blockspergrid_x = math.ceil(n_h/threadsperblock[0])
        blockspergrid_y = math.ceil(n_h/threadsperblock[1])
        blockspergrid = (blockspergrid_x, blockspergrid_y)

        cuda_kernal[blockspergrid, threadsperblock](d_meanvec, d_dot_of_selected, candidate,
                                                    d_covariance, self.grid_priori[0][0],
                                                    d_lookup_table, d_results)

        summation = sum_reduce(d_results)

        return 1 - summation

    def o_t_approx_host_old(self, subset_index, cuda_kernal):
        '''host code for o_t_approx.
        Args:
            subset_index (np.ndarray, n=1): index of some sensors
            cuda_kernal (cuda_kernals.o_t_approx_kernal or o_t_approx_dist_kernal)
        Return:
            (float): o_t_approx
        '''
        n_h = len(self.transmitters)  # number of hypotheses/transmitters
        sub_cov = self.covariance_sub(subset_index)
        sub_cov_inv = np.linalg.inv(sub_cov)  # inverse
        d_meanvec_array = cuda.to_device(self.meanvec_array)
        d_subset_index = cuda.to_device(subset_index)
        d_sub_cov_inv = cuda.to_device(sub_cov_inv)
        d_results = cuda.device_array((n_h, n_h), np.float64)

        threadsperblock = (self.TPB, self.TPB)
        blockspergrid_x = math.ceil(n_h / threadsperblock[0])
        blockspergrid_y = math.ceil(n_h / threadsperblock[1])
        blockspergrid = (blockspergrid_x, blockspergrid_y)
        priori = self.grid_priori[0][0]  # priori is uniform, equal everywhere

        cuda_kernal[blockspergrid, threadsperblock](d_meanvec_array, d_subset_index, d_sub_cov_inv, priori, d_results)

        results = d_results.copy_to_host()
        # print_results(results)
        return 1 - results.sum()

    def update_dot_of_selected_host(self, d_dot_of_selected, best_candidate, d_covariance, d_meanvec):
        '''Host code for updating dot_of_selected after a new sensor is seleted
           TYPE = "numba.cuda.cudadrv.devicearray.DeviceNDArray", which cannot be pickled --> cannot exist before using joblib
        Args:
            d_dot_of_selected (TYPE): stores the np.dot(np.dot(pj_pi, sub_cov_inv), pj_pi)) of sensors already selected
                                      in previous iterations. shape=(m, m) where m is the number of hypothesis (grid_len^2)
            best_candidate (int)    : the best candidate sensor selected in the iteration
            d_covariance (TYPE)     : covariance matrix
            d_meanvec (TYPE)        : contains the mean vector of every transmitter
        '''
        n_h = len(self.transmitters)
        threadsperblock = (self.TPB, self.TPB)
        blockspergrid_x = math.ceil(n_h/threadsperblock[0])
        blockspergrid_y = math.ceil(n_h/threadsperblock[1])
        blockspergrid = (blockspergrid_x, blockspergrid_y)

        update_dot_of_selected_kernal[blockspergrid, threadsperblock](d_meanvec, d_dot_of_selected, best_candidate, d_covariance)


    #@profile
    def o_t_host(self, subset_index):
        '''host code for o_t.
        Args:
            subset_index (np.ndarray, n=1): index of some sensors
        '''
        n_h = len(self.transmitters)   # number of hypotheses/transmitters
        sub_cov = self.covariance_sub(subset_index)
        sub_cov_inv = np.linalg.inv(sub_cov)           # inverse
        d_meanvec_array = cuda.to_device(self.meanvec_array)
        d_subset_index = cuda.to_device(subset_index)
        d_sub_cov_inv = cuda.to_device(sub_cov_inv)
        d_results = cuda.device_array((n_h, n_h), np.float64)
        d_lookup_table    = cuda.to_device(self.lookup_table_q)

        threadsperblock = (self.TPB, self.TPB)
        blockspergrid_x = math.ceil(n_h/threadsperblock[0])
        blockspergrid_y = math.ceil(n_h/threadsperblock[1])
        blockspergrid = (blockspergrid_x, blockspergrid_y)

        o_t_kernal[blockspergrid, threadsperblock](d_meanvec_array, d_subset_index, d_sub_cov_inv, d_lookup_table, d_results)

        results = d_results.copy_to_host()
        return np.sum(results.prod(axis=1)*self.grid_priori[0][0])


    # @profile
    def o_t_host_iter(self, d_dot_of_selected, candidate, d_covariance, d_meanvec, d_results, cuda_kernal, d_lookup_table):
        '''Host code for o_t
           The iteration version of o_t_host. Iteration suggests the current iteration uses results from the previous iterations
           When iteration technique is used, O(B^2) time is reduced to O(1). Same iteration idea is used in o_t_approx_host.
            TYPE = "numba.cuda.cudadrv.devicearray.DeviceNDArray", which cannot be pickled --> cannot exist before using joblib
        Args:
            d_dot_of_selected (TYPE): stores the np.dot(np.dot(pj_pi, sub_cov_inv), pj_pi)) of sensors already selected
                                      in previous iterations. shape=(m, m) where m is the number of hypothesis (grid_len^2)
            candidate (int)         : a candidate sensor index
            d_covariance (TYPE)     : covariance matrix
            d_meanvec (TYPE)        : contains the mean vector of every transmitter
            d_results (TYPE)        : 2D array. save the results for each (i, j) pair of transmitter and sensor's error
            cuda_kernal (cuda_kernals.o_t_approx_kernal2 or o_t_approx_dist_kernal2)
            d_lookup_table (TYPE)   : trade space for time

        '''
        n_h = len(self.transmitters)
        threadsperblock = (self.TPB, self.TPB)
        blockspergrid_x = math.ceil(n_h/threadsperblock[0])
        blockspergrid_y = math.ceil(n_h/threadsperblock[1])
        blockspergrid = (blockspergrid_x, blockspergrid_y)

        cuda_kernal[blockspergrid, threadsperblock](d_meanvec, d_dot_of_selected, candidate, d_covariance, d_lookup_table, d_results)
        
        results = d_results.copy_to_host()
        return np.sum(results.prod(axis=1)*self.grid_priori[0][0])


    def convert_to_pos(self, true_indices):
        list = []
        for index in true_indices:
            x = index // self.grid_len
            y = index % self.grid_len
            list.append((x, y))
        return list

    def select_offline_greedy_lazy_old(self, budget, cores, cuda_kernal):
        '''(Parallel + Lazy greedy) Select a subset of sensors greedily. offline + homo version using ** GPU **
        Args:
            budget (int): budget constraint
            cores (int): number of cores for parallelzation
            cuda_kernal (cuda_kernals.o_t_approx_kernal or o_t_approx_dist_kernal): the O_{aux} in the paper
        Return:
            (list): an element is [str, int, float],
                    where str is the list of subset_index, int is # of sensors, float is O_T
        '''
        counter = 0
        base_ot_approx = 0
        if cuda_kernal == o_t_approx_kernal:
            base_ot_approx = 1 - 0.5 * len(self.transmitters)
        elif cuda_kernal == o_t_approx_dist_kernal:
            largest_dist = (self.grid_len - 1) * math.sqrt(2)
            max_gain_up_bound = 0.5 * len(self.transmitters) * largest_dist  # the default bound is for non-distance
            for sensor in self.sensors:  # need to update the max gain upper bound for o_t_approx with distance
                sensor.gain_up_bound = max_gain_up_bound
            base_ot_approx = (1 - 0.5 * len(self.transmitters)) * largest_dist

        plot_data = []
        cost = 0  # |T| in the paper
        subset_index = []  # T   in the paper
        complement_sensors = copy.deepcopy(self.sensors)  # S\T in the paper
        subset_to_compute = []
        while cost < budget and complement_sensors:
            best_candidate = complement_sensors[0].index  # init as the first sensor
            best_sensor = complement_sensors[0]
            complement_sensors.sort()  # sorting the gain descendingly
            new_base_ot_approx = 0
            # for sensor in complement_sensors:
            #    print((sensor.index, sensor.gain_up_bound), end=' ')
            update, max_gain = 0, 0
            while update < len(complement_sensors):
                update_end = update + cores if update + cores <= len(complement_sensors) else len(complement_sensors)
                candidiate_index = []
                for i in range(update, update_end):
                    candidiate_index.append(complement_sensors[i].index)
                counter += 1
                candidate_results = Parallel(n_jobs=cores)(
                    delayed(self.inner_greedy)(subset_index, cuda_kernal, candidate) for candidate in candidiate_index)
                # an element of candidate_results is a tuple - (index, o_t_approx, subsetlist)
                for i, j in zip(range(update, update_end), range(0,
                                                                 cores)):  # the two range might be different, if the case, follow the first range
                    complement_sensors[i].gain_up_bound = candidate_results[j][
                                                              1] - base_ot_approx  # update the upper bound of gain
                    # print(candidate_results[j][2], candidate_results[j][1], base_ot_approx, complement_sensors[i].gain_up_bound)
                    if complement_sensors[i].gain_up_bound > max_gain:
                        max_gain = complement_sensors[i].gain_up_bound
                        best_candidate = candidate_results[j][0]
                        best_sensor = complement_sensors[i]
                        new_base_ot_approx = candidate_results[j][1]

                if update_end < len(complement_sensors) and max_gain > complement_sensors[
                    update_end].gain_up_bound:  # where the lazy happens
                    # print('\n***LAZY!***\n', cost, (update, update_end), len(complement_sensors), '\n')
                    break
                update += cores
            base_ot_approx = new_base_ot_approx  # update the base o_t_approx for the next iteration
            print(best_candidate, subset_index, base_ot_approx, '\n')
            ordered_insert(subset_index, best_candidate)  # guarantee subset_index always be sorted here
            subset_to_compute.append(copy.deepcopy(subset_index))
            plot_data.append([len(subset_index), base_ot_approx,
                              0])  # don't compute real o_t now, delay to after all the subsets are selected
            complement_sensors.remove(best_sensor)
            if base_ot_approx > 0.9999999999999:
                break
            cost += 1
        print('number of o_t_approx', counter)
        subset_results = Parallel(n_jobs=cores)(
            delayed(self.inner_greedy_real_ot_old)(subset_index) for subset_index in subset_to_compute)

        for i in range(len(subset_results)):
            plot_data[i][2] = subset_results[i]

        return plot_data


if __name__ == '__main__':
    pass
