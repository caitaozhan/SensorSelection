import pandas as pd
from datetime import datetime
import numpy as np
from os import walk
import matplotlib.pyplot as plt
import scipy

import os
# class Utilities:
#     peakValues = None
#     numOfPoints = 0
#     meanDict = {}
#     varianceDict = {}
#
#     def totalpower(self, psd):
#         avg_psd_dB = 10 * np.log10(np.average(psd) / 10.0)
#         return avg_psd_dB
#
#     def get_peak_pos(self, reals, imags, sample_rate, fc, NFFT):
#         numFFTs = int(len(reals) / NFFT)
#         maxValues = np.zeros(numFFTs)
#         averageValues = np.zeros(numFFTs)
#         positions = np.zeros(numFFTs)
#         errors = 0
#
#         i = 0
#         iq_samples = np.array(
#             [(re + 1j * co) for re, co in zip(reals[i * NFFT:(i + 1) * NFFT], imags[i * NFFT:(i + 1) * NFFT])])
#         x = plt.psd(iq_samples, NFFT=NFFT, Fs=sample_rate / 1e6, Fc=fc)
#         #print(iq_samples, NFFT, sample_rate / 1e6, fc)
#         np.set_printoptions(threshold=np.infty)
#         #print(x[0])
#         firsthalf = 10 * np.log10(x[0][0:int(1 * len(x[0]) / 2) - 1])
#         peakPoint = np.argmax(firsthalf)
#         return peakPoint
#
#     def collect_peaks(self, reals, imags, sample_rate, fc, NFFT, peakPoint):
#         numFFTs = 1
#         maxValues = np.zeros(numFFTs)
#         averageValues = np.zeros(numFFTs)
#         positions = np.zeros(numFFTs)
#         errors = 0
#         for i in range(0, numFFTs):  # len(reals)/NFFT
#
#             iq_samples = np.array(
#                 [(re + 1j * co) for re, co in zip(reals[i * NFFT:(i + 1) * NFFT], imags[i * NFFT:(i + 1) * NFFT])])
#             x = plt.psd(iq_samples, NFFT=NFFT, Fs=sample_rate / 1e6, Fc=fc)
#
#             np.set_printoptions(threshold=np.infty)
#
#             firsthalf = 10 * np.log10(x[0][0:int(1 * len(x[0]) / 2) - 1])
#             self.peakValues[self.numOfPoints] = firsthalf[peakPoint]
#             self.numOfPoints += 1
#
#     def post_process(self, filename):
#         mean = np.mean(self.peakValues[:self.numOfPoints])
#         variance = np.var(self.peakValues[:self.numOfPoints])
#         std = np.std(self.peakValues[:self.numOfPoints])
#         self.meanDict[filename] = mean
#         self.varianceDict[filename] = variance
#
#     def __init__(self):
#         self.peakValues = np.zeros(256)
#
# class RTL_IQ_analysis:
#     def __init__(self, iqfile, datatype, block_length, sample_rate):
#         self.iqfile = iqfile
#         self.datatype = datatype
#         self.sizeof_data = self.datatype.nbytes  # number of bytes per sample in file
#         self.block_length = block_length
#         self.sample_rate = sample_rate
#         self.hfile = open(self.iqfile, "rb")
#         self.call_count = 0
#
#         def on_die(killed_ref):
#             print('on_die')
#             self.hfile.close()

class Utilities:

    def totalpower(self, psd):

        avg_psd_dB = 10 * np.log10(np.average(psd) / 10.0)
        return avg_psd_dB

    def plot_fft(self, reals, imags, sample_rate, fc, NFFT):
        numFFTs = 1
        maxValues = np.zeros(numFFTs)
        averageValues = np.zeros(numFFTs)
        positions = np.zeros(numFFTs)
        errors = 0
        for i in range(0, numFFTs): #len(reals)/NFFT

            iq_samples = np.array([(re + 1j * co) for re, co in zip(reals[i * NFFT:(i + 1) * NFFT], imags[i * NFFT:(i + 1) * NFFT])])
            x = plt.psd(iq_samples, NFFT=NFFT, Fs=sample_rate / 1e6, Fc=fc)

            np.set_printoptions(threshold=np.infty)
            firsthalf = 10 * np.log10(x[0][0:int(1 * len(x[0]) / 2) - 1])
            print(np.max(firsthalf))
            #print(positions[i])
        #print (positions)
        return np.max(firsthalf)


import weakref
class RTL_IQ_analysis:
    def __init__(self, iqfile, datatype, block_length, sample_rate):
        self.iqfile = iqfile
        self.datatype = datatype
        self.sizeof_data = self.datatype.nbytes  # number of bytes per sample in file
        self.block_length = block_length
        self.sample_rate = sample_rate
        self.hfile = open(self.iqfile, "rb")
        self.call_count = 0
        def on_die(killed_ref):
            print('on_die')
            self.hfile.close()
        self._del_ref = weakref.ref(self, on_die)

    #i / (255 / 2) - 1, q / (255 / 2) - 1
    def read_samples(self):
        self.hfile.seek(self.block_length * self.call_count)
        self.call_count += 1
        try:
            iq = scipy.fromfile(self.hfile, dtype=self.datatype, count=self.block_length, )
        except MemoryError:
            print("End of File")
        else:
            reals = scipy.array([(r / (255.0 / 2) - 1) for index, r in enumerate(iq) if index % 2 == 0])
            imags = scipy.array([(i / (255.0 / 2) - 1) for index, i in enumerate(iq) if index % 2 == 1])

        # self.hfile.close()
        return reals, imags


def parse_gps_time(filename):
    gpgga_df = pd.read_csv(filename, sep=',')
    hours = gpgga_df['1'].values // 10000
    minutes = (gpgga_df['1'].values - (hours * 10000)) // 100
    seconds = gpgga_df['1'].values % 100
    #hours -= 4

    timestamp = [datetime(2019, 7, 28, int(hours[i]), int(minutes[i]), int(seconds[i])) for i in range(len(hours))]
    epoch = datetime(1970, 1, 1)
    seconds_list = np.array([int((t - epoch).total_seconds()) for t in timestamp])
    gpgga_df['seconds'] = seconds_list
    #print(seconds_list)
    return gpgga_df
    #print(timestamp)

def process_iq(filename, NFFT = 128):
    datatype = scipy.uint8
    block_length = NFFT
    # block_offset = NFFT*i #<---change to random offsets between 0 to (max_no_of_iq_samples - block_length)
    sample_rate = 2e6
    fc = 916e6
    utils = Utilities()

    # fullFileName = "/" + str(filename) +".iq"
    # print(fullFileName)
    rtl = RTL_IQ_analysis(filename, datatype, block_length, sample_rate)
    for j in range(1):
        r, i = rtl.read_samples()
        # print (r,i)
        peakPower = np.array([utils.plot_fft(r, i, sample_rate, fc, NFFT)])
    return peakPower


def determine_relevant_timestamp(gpgga_df):
    mypath = '/home/arani2/misc-work/testbed-southP'
    iq_dfs = [[0] * 10] * 10

    sensor_file = mypath + '/hosts_and_addresses.txt'
    sensor_df = pd.read_csv(sensor_file, sep=' ', header=None, usecols=[4, 6, 7], names=['dir_name', 'x', 'y'])
    sensor_df['dir_name'] = sensor_df['dir_name'].str[0:-4]

    temp_df = pd.DataFrame(columns=['i', 'j', 'sensor_loc_x', 'sensor_loc_y', 'mean', 'std'])
    index = 0
    for i in range(10):
        for j in range(10):
            relevant_df = gpgga_df[(gpgga_df['x_coords'] == i) & (gpgga_df['y_coords'] == j)]
            #print(i, j, relevant_df.tail(1))
            dir_list = os.listdir(mypath)
            for dir in dir_list:
                if dir[0:2] == 'o-' or dir[0:2] == 'r-':
                #if dir == 'o-103':
                    #print(dir)
                    iq_file_list = os.listdir(mypath + '/' + dir)
                    file_timestamp = [int(iq_file[3:-6]) for iq_file in iq_file_list if len(iq_file) > 14]
                    iq_dfs[i][j] = pd.DataFrame({'filename': file_timestamp})
                    #print(iq_dfs[i][j])
                    merged_df = pd.merge_asof(relevant_df, iq_dfs[i][j], left_on='seconds', right_on='filename', direction='nearest')
                    try:
                        relevant_file = merged_df.head(1)['filename'].values[0]
                        filename = str('iq-' + str(relevant_file) + '.trial')
                        print(i, j, str('iq-' + str(relevant_file) + '.trial'))
                        peak_values = process_iq(mypath + '/' + dir + '/' + filename)
                        sensor_loc_x = sensor_df[sensor_df['dir_name'] == dir]['x'].values[0]
                        sensor_loc_y = sensor_df[sensor_df['dir_name'] == dir]['y'].values[0]
                        temp_df.loc[index] = [i, j, sensor_loc_x, sensor_loc_y, np.mean(peak_values), np.std(peak_values)]
                        index += 1
                    except Exception as e:
                        print('Exception', e)
                        pass
    temp_df['std'] = 1
    temp_df['std'] = temp_df.groupby(['sensor_loc_x', 'sensor_loc_y'])['std'].transform('mean')

    temp_df['i'] = temp_df['i'].values.astype(int)
    temp_df['j'] = temp_df['j'].values.astype(int)
    temp_df['sensor_loc_x'] = temp_df['sensor_loc_x'].values.astype(int)
    temp_df['sensor_loc_y'] = temp_df['sensor_loc_y'].values.astype(int)

    temp_df.to_csv('hypothesis', sep=' ', header=None, index=False)
    print(temp_df)
    sensor_df = temp_df[['sensor_loc_x', 'sensor_loc_y', 'std']].drop_duplicates(['sensor_loc_x', 'sensor_loc_y'])
    cov_df = np.square(sensor_df['std'].values)

    cov_df = np.diag(cov_df)
    cov_file = open('cov', 'w')
    np.savetxt(cov_file, cov_df, fmt='%.4f')
    cov_file.close()
    sensor_df['cost'] = 1
    sensor_df.to_csv('sensor', header=None, index=False, sep=' ')


    #print(cov_df)

gpgga_df = parse_gps_time('filtered_tran.csv')
determine_relevant_timestamp(gpgga_df)
