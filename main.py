# This file is an example of usage of utils functions, the code is for analyzing the effective connectivity between two channels of recorded EEG signals by mean of Granger Causality.

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

from utils import *


EEG = sio.loadmat(r'F:\HWs\MSc\Research\Depression Dataset\Testing Preprocess\567\567_EpochedON_Positive_Feedbacks_2B3A.set')

EEG_data = EEG['data']
ERPed_Data = []
ERPed_Data.append(np.mean(EEG_data[:, :, :18], axis = 2))
ERPed_Data.append(np.mean(EEG_data[:, :, 18:36], axis = 2))

print(len(ERPed_Data), ERPed_Data[0].shape)
print(EEG_data.shape)

file_loc = r'F:\HWs\MSc\Research\Data Process\Preprocessing\New Approach\chanlocs_py.txt'

channel_name = channels_specs(file_loc)
print(channel_name)

channels = np.array([1, 3, 5, 6, 9, 12])

# orders_mat = []

print("Order Estimation")
orders_mat = OrderEstimate_byChannels((ERPed_Data[0] + ERPed_Data[1]) / 2, channels, 200, 50, 4)

window_length = 500
overlap_ratio = 0.94

GC_tests = []

print("Block 1")
GC_tests.append(GrangerCausalityEstimator(ERPed_Data[0], channels, window_length, overlap_ratio, orders_mat))

print("Block 2")
GC_tests.append(GrangerCausalityEstimator(ERPed_Data[1], channels, window_length, overlap_ratio, orders_mat))

t = np.arange(-2, 3, 5 / len(GC_tests[0][:, 0, 0]))

for c in range(len(channels)):
    
    for i in range(len(channels)):
        
        if i != c:
            
            plt.figure(figsize = (20, 5))
            plt.subplot(1, 2, 1)
            plt.plot(t, GC_tests[0][:, c, i], label = str(channels[i]))

            # plt.legend()
            plt.axvline(x = 0, color = 'k')
            plt.xlim([t[0], t[-1]])
            # plt.ylim([0.1, 0.35])
            plt.title("Channel " + str(channel_name[channels[i]][:3]) + " to channel " + str(channel_name[channels[c]][:3]) + " Granger Causality Values in BLock 1 " + data_exp)
            plt.ylabel("GC values")
            plt.xlabel("time (s)")
            
            plt.subplot(1, 2, 2)
            plt.plot(t, GC_tests[1][:, c, i], label = str(channels[i]))

            # plt.legend()
            plt.axvline(x = 0, color = 'k')
            plt.xlim([t[0], t[-1]])
            # plt.ylim([0.1, 0.35])
            plt.title("Channel " + str(channel_name[channels[i]][:3]) + " to channel " + str(channel_name[channels[c]][:3]) + " Granger Causality Values in BLock 2 " + data_exp)
            plt.ylabel("GC values")
            plt.xlabel("time (s)")
            plt.savefig(r"F:\HWs\MSc\Research\Data Process\Preprocessing\PAC_init_analysis\Results\Subject_" + str(subject) + "\\s" + str(subject) + locking_type + "_GrangerCausalityOfChannel " + str(channel_name[channels[i]][:3]) + " to channel " + str(channel_name[channels[c]][:3]) + data_exp)
            plt.show()
