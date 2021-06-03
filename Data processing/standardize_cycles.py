import numpy as np
import sys
import pandas as pd
from pandas.core.common import flatten
import matplotlib.pyplot as plt
import csv
from scipy.signal import savgol_filter,find_peaks,argrelmax
from scipy.interpolate import CubicSpline
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from time import time 
import operator
import math
from numpy import savetxt
from operator import itemgetter 
np.set_printoptions(threshold=np.inf)


ids_pre_cycle = [17.0,51.0,59.0,67.0,93.0,144.0,195.0,217.0,227.0,231.0,278.0,367.0,371.0,468.0,503.0,541.0,556.0,570.0,578.0,647.0,711.0,839.0,890.0,905.0,967.0]
ids_mid_cycle = [17.0,51.0,59.0,67.0,93.0,144.0,195.0,217.0,227.0,231.0,278.0,367.0,371.0,468.0,503.0,541.0,556.0,570.0,578.0,647.0,711.0,839.0,890.0,905.0,967.0]
ids_post_cycle = [17.0,51.0,59.0,67.0,144.0,195.0,217.0,231.0,278.0,367.0,468.0,503.0,541.0,556.0,570.0,578.0,647.0,839.0,890.0,905.0,967.0]

count = 0 
#for s in ids_pre_cycle:
#for s in ids_mid_cycle:
for s in ids_post_cycle:
    print(s)
    # Returns raw_time, pressure, dbp, sbp
    def getSignal(pre_mid_post, pat_id):
        full_data_set = pd.read_csv('Data/data_set_without_cycles.csv', header=0)
        if pre_mid_post == 'pre':
            raw_time = pd.read_csv('Data/FP_pre/raw_time_pat_'+str(pat_id)+'.csv', header=0)
            pressure = pd.read_csv('Data/FP_pre/pressure_pat_'+str(pat_id)+'.csv', header=0)
            if not (math.isnan(full_data_set.clinical_visits_pre_24h_sbp_mean_value[full_data_set['roottable_case_id_text'] == pat_id].values)):
                sbp = int(full_data_set.clinical_visits_pre_24h_sbp_mean_value[full_data_set['roottable_case_id_text'] == pat_id])
            else:
                sbp = None
            if not (math.isnan(full_data_set.clinical_visits_pre_24h_dbp_mean_value[full_data_set['roottable_case_id_text'] == pat_id].values)):
                dbp = int(full_data_set.clinical_visits_pre_24h_dbp_mean_value[full_data_set['roottable_case_id_text'] == pat_id])
            else:
                dbp = None
        elif pre_mid_post == 'mid':
            raw_time = pd.read_csv('Data/FP_mid/raw_time_pat_'+str(pat_id)+'_mid.csv', header=0)
            pressure = pd.read_csv('Data/FP_mid/pressure_pat_'+str(pat_id)+'_mid.csv', header=0)
            if not (math.isnan(full_data_set.clinical_visits_mid_24h_sbp_mean_value[full_data_set['roottable_case_id_text'] == pat_id].values)):
                sbp = int(full_data_set.clinical_visits_mid_24h_sbp_mean_value[full_data_set['roottable_case_id_text'] == pat_id])
            else:
                sbp = None
            if not (math.isnan(full_data_set.clinical_visits_mid_24h_dbp_mean_value[full_data_set['roottable_case_id_text'] == pat_id].values)):
                dbp = int(full_data_set.clinical_visits_mid_24h_dbp_mean_value[full_data_set['roottable_case_id_text'] == pat_id])
            else:
                dbp = None
        elif pre_mid_post == 'post':
            raw_time = pd.read_csv('Data/FP_post/raw_time_pat_'+str(pat_id)+'_post.csv', header=0)
            pressure = pd.read_csv('Data/FP_post/pressure_pat_'+str(pat_id)+'_post.csv', header=0)
            if not (math.isnan(full_data_set.clinical_visits_post_24h_sbp_mean_value[full_data_set['roottable_case_id_text'] == pat_id].values)):
                sbp = int(full_data_set.clinical_visits_post_24h_sbp_mean_value[full_data_set['roottable_case_id_text'] == pat_id])
            else:
                sbp = None
            if not (math.isnan(full_data_set.clinical_visits_post_24h_dbp_mean_value[full_data_set['roottable_case_id_text'] == pat_id].values)):
                dbp = int(full_data_set.clinical_visits_post_24h_dbp_mean_value[full_data_set['roottable_case_id_text'] == pat_id])
            else:
                dbp = None
    
        return raw_time, pressure, dbp, sbp

    raw_time, pressure, dbp, sbp = getSignal('post',s)
    # Plot original pressure signal
    '''
    if s == 647:
        fig, axs = plt.subplots(3)
        fig.tight_layout()
        axs[0].plot(raw_time, pressure, color='midnightblue')
        axs[1].plot(raw_time[3000:15000], pressure[3000:15000], color='midnightblue')
        axs[2].plot(raw_time[51200:52200], pressure[51200:52200], color='midnightblue')
        axs[2].set_xlabel('Time Points [-]', size='x-large')
        axs[1].set_ylabel('Blood Pressure [mmHg]',size='x-large')
        plt.show()
        quit()
    else:
        continue
    '''
    
    # Returns cycles_times, cycles_pressures
    def split_signal(raw_time, pressure):
        pressure = pressure.T
        raw_time = raw_time.T
        raw_time = raw_time.to_numpy()
        pressure = pressure.to_numpy()
        pressure = pressure.reshape(-1)
        raw_time = raw_time.reshape(-1)
        peaks, _ = find_peaks(-1*pressure, distance=150)

        cycles_times = []
        cycles_pressures = []
        for i in range(1,len(peaks)):
            times = []
            pressures = []
            for j in range(peaks[i-1],peaks[i]):
                times.append(raw_time[j])
                pressures.append(pressure[j])
            cycles_times.append(times)
            cycles_pressures.append(pressures)

        # Plot peaks
        '''
        plt.plot(pressure)
        #plt.plot(peaks, pressure[peaks], "x")
        #plt.plot(np.zeros_like(pressure), "--", color="gray")
        plt.show()
        '''

        return cycles_times, cycles_pressures

    cycles_times, cycles_pressures = split_signal(raw_time, pressure)

    # Returns pressures_sav
    def savitzky_golay(cycles, coarseness):
        pressures_sav = []
        for e in cycles:
            sav = savgol_filter(e, coarseness, 3) # parameter 2: how coarse the smoothing is
            pressures_sav.append(sav)
        # Plot smoothing
        '''
        if s == 647:
            for i in range(len(cycles_pressures)):
                plt.plot(cycles_pressures[i], color='midnightblue',label='Original Cycle')
                plt.plot(pressures_sav[i], color='darkorange',label='Savitsky Golay Filter')
                plt.xlabel('Time Points [-]')
                plt.ylabel('Blood Pressure [mmHg]')
                plt.legend(bbox_to_anchor=(1,1))
                plt.show()
            quit()
        '''
        
        return pressures_sav

    cycles_sav = savitzky_golay(cycles_pressures, 21)

    # Returns pressures_stand
    def scale_x(cycles, cycles_times):
        pressures_stand = []
        for i in range(len(cycles)):
            stand = []
            start = cycles_times[i][0]
            end = cycles_times[i][-1]
            for n in range(0,100,1):
                pressure = np.interp(start+n/100*(end-start),cycles_times[i],cycles[i])
                stand.append(pressure)
            pressures_stand.append(stand)
        # Plot scaled curve
        '''
        if s == 647:
            for i in range(len(pressures_stand)):
                plt.plot(cycles[i], 'midnightblue', label='Original Cycle')
                plt.plot(pressures_stand[i], 'darkorange',label='Scaled Cycle')
                plt.xlabel('Time Points [-]')
                plt.ylabel('Blood Pressure [mmHg]')
                plt.legend(bbox_to_anchor=(1,1))
                plt.show()
            quit()
        
        '''
        return pressures_stand

    cycles_stand = scale_x(cycles_sav,cycles_times) 

    # Returns pressures_lim
    def limit_cycles(cycles, dbp_min, sbp_min, sbp_min_i, sbp_max_i):
        pressures_lim = []
        for i in range(len(cycles)):
            if np.min(cycles[i]) > dbp_min and np.max(cycles[i]) > sbp_min:
                index, value = max(enumerate(cycles[i]), key=operator.itemgetter(1))
                if index > sbp_min_i and index < sbp_max_i:
                    for w in np.arange(0,len(cycles[i])-29,10):
                        dev = np.std(cycles[i][w:w+30])
                        if dev < 0.6:
                            break
                    pressures_lim.append(cycles[i])

        return pressures_lim

    cycles_lim = limit_cycles(cycles_stand,40,90,14,31)

    # Returns trend, stable
    def find_trend(cycles):
        signal = list(flatten(cycles))
        trend = savgol_filter(signal, 1001, 3) # parameter 2: how coarse the smoothing is
        stable = signal - trend

        return trend, stable

    # Returns new_cycles
    def getCycles_without_trend(cycles):
        trend, stable = find_trend(cycles)
        new_cycles = []
        for n in np.arange(0,len(stable),100):
            cycle = []
            for i in range(100):
                cycle.append(stable[n+i])
            new_cycles.append(cycle)
        
        return new_cycles

    cycles_detrended = getCycles_without_trend(cycles_lim)

    # Returns arrays
    def best_window(cycles_detrended):
        min_dev = math.inf
        min_w = 0
        for w in range(len(cycles_detrended)-8):
            a = list(flatten(cycles_detrended[w:w+8]))
            deviation = 0
            for p in np.arange(0,100,10):
                points = []
                for c in range(8):
                    points.append(a[c*100+p])
                dev = np.std(points)
                deviation += dev
            if deviation < min_dev:
                min_dev = deviation
                min_w = w

        # Plot best window
        '''
        fig, axs = plt.subplots(2)
        fig.tight_layout()
        axs[0].plot(list(flatten(cycles_detrended)),color='midnightblue', label='Full Signal')
        axs[0].plot(np.arange(list(flatten(cycles_detrended)).index(list(flatten(cycles_detrended[min_w]))[0]),list(flatten(cycles_detrended)).index(list(flatten(cycles_detrended[min_w+7]))[-1])+1,1),list(flatten(cycles_detrended[min_w:min_w+8])), color='darkorange',label='Best Window')
        axs[1].plot(list(flatten(cycles_detrended)),color='midnightblue')
        axs[1].plot(np.arange(list(flatten(cycles_detrended)).index(list(flatten(cycles_detrended[min_w]))[0]),list(flatten(cycles_detrended)).index(list(flatten(cycles_detrended[min_w+7]))[-1])+1,1),list(flatten(cycles_detrended[min_w:min_w+8])), color='darkorange')
        axs[1].set_xlabel('Time Points [-]', size = 'large')
        axs[0].set_ylabel('Blood Pressure [mmHg]', size='large')
        axs[1].set_ylabel('Blood Pressure [mmHg]', size='large')
        axs[0].legend()#bbox_to_anchor=(1,1))
        plt.show()
        '''
        
        arrays = [list(x) for x in cycles_lim[min_w:min_w+8]]
        
        return arrays, min_w

    arrays,min_w = best_window(cycles_detrended)

    # Returns cycles
    def split_window(arrays):
        cycles = []
        for n in range(len(arrays)):
            cyc = []
            for i in range(100):
                cyc.append(arrays[n][i])
            cycles.append(cyc)

        return cycles

    # Returns rep_cycle
    def find_representative_cycle(arrays):
        cycles_window = split_window(arrays)
        mean_curve = [np.mean(k) for k in zip(*arrays)]
        min_avvik = np.inf
        min_index = 0
        for c in range(len(cycles_window)):
            dev = 0
            for p in range(0,100,10):
                sa = abs(cycles_window[c][p] - mean_curve[p])
                dev += sa
            if dev < min_avvik:
                min_avvik = dev
                min_index = c
        rep_cycle = cycles_window[min_index]

        return rep_cycle
    
    mean_cycle = find_representative_cycle(arrays)

    # Returns scaled_cycle
    def scale_y(cycle, dbp, sbp):
        scaled_cycle = []
        peak = np.max(cycle)
        bottom = np.min(cycle)
        if sbp != np.nan and dbp != np.nan:
            for j in range(0, len(cycle), 1):
                y = cycle[j]
                norm_dist_from_bottom = (y-bottom)/(peak-bottom)
                norm_dist_from_top = 1-norm_dist_from_bottom
                new_y = sbp * norm_dist_from_bottom + dbp * norm_dist_from_top
                scaled_cycle.append(new_y)
        
        plt.plot(cycle, color='midnightblue', label='Original Cycle')
        plt.plot(scaled_cycle, color='darkorange', label='Scaled Cycle')
        plt.axhline(y = sbp, color='black', label='Ambulatory SBP')
        plt.axhline(y = dbp, color='black', label='Ambulatory DBP')
        plt.ylabel('Blood Pressure [mmHg]')
        plt.xlabel('Time Points [-]')
        plt.legend(bbox_to_anchor=(1,0.91))
        plt.show()
        
        
        return scaled_cycle
  
    if not (dbp == None and sbp == None):
        rep_cycle = scale_y(mean_cycle, dbp, sbp)

        #savetxt('Data/FP_pre_cycle/Pre_finger_pressure_patient_' + str(s) + '.csv', rep_cycle, delimiter=',')
        #savetxt('Data/FP_mid_cycle/Mid_finger_pressure_patient_' + str(s) + '.csv', rep_cycle, delimiter=',')
        #savetxt('Data/FP_post_cycle/Post_finger_pressure_patient_' + str(s) + '.csv', rep_cycle, delimiter=',')
    else: 
        print(s, ' No ambulatory')
    print(s, 'done')

