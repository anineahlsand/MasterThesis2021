import pandas as pd
import numpy as np
from numpy import savetxt
from matplotlib import pyplot as plt
import csv
from ast import literal_eval
import warnings
from statistics import mean

def getData():
    flow_data = pd.read_csv('Data/ts_export_20210326.csv')
    flow_data = flow_data.iloc[:,2:]
    flow_data.drop(['test_day','acq_time','synced_time','units','MyMDTLabel','mark', 'synced_dicom', 'sync_offset'], axis=1, inplace=True)
    return flow_data

flow_data = getData()

def getValueData():
    value_data = pd.read_csv("Data/Echodata_KA_Trial.csv",delimiter=';')
    value_data = value_data[['roottable_case_id_text', 'clinical_visits_test_day_item','SV(4D)', 'LVOT SV']]
    value_data.columns = ['roottable_case_id_text', 'clinical_visits_test_day_item','SV', 'LVOT_SV']
    value_data['roottable_case_id_text'] = pd.to_numeric(value_data['roottable_case_id_text'])
    value_data['SV'] = value_data['SV'].str.replace(',','.')
    value_data['LVOT_SV'] = value_data['LVOT_SV'].str.replace(',','.')

    return value_data

value_data = getValueData()

def find_SV(patientId):
    pre_SV, mid_SV, post_SV= 0,0,0
    for i in range(len(value_data.roottable_case_id_text.values)):
        if value_data.roottable_case_id_text.values[i] == patientId:
            if value_data.clinical_visits_test_day_item.values[i] == 'Pre-test day 2':
                if value_data.SV.values[i] == '#VERDI!':
                    pre_SV = float(value_data.LVOT_SV.values[i])
                elif pd.isnull(value_data.SV.values[i]):
                    pre_SV = 0
                else:
                    pre_SV = float(value_data.SV.values[i])
            elif value_data.clinical_visits_test_day_item.values[i] == 'Mid-test':
                if value_data.SV.values[i] == '#VERDI!':
                    mid_SV = float(value_data.LVOT_SV.values[i])
                elif pd.isnull(value_data.SV.values[i]):
                    mid_SV = 0
                else:
                    mid_SV = float(value_data.SV.values[i])
            elif value_data.clinical_visits_test_day_item.values[i] == 'Post-test day 2':
                if value_data.SV.values[i] == '#VERDI!':
                    post_SV = float(value_data.LVOT_SV.values[i])
                elif pd.isnull(value_data.SV.values[i]):
                    post_SV = 0
                else:
                    post_SV = float(value_data.SV.values[i])
    return pre_SV, mid_SV, post_SV

def findIds(df):
    ids = []
    for i in df.trial_id.values:
        if i not in ids:
            ids.append(i)
    return ids

def findPais(df):
    pais = []
    for i in df.patient.values:
        if i not in pais:
            pais.append(i)
    return pais

ids = findIds(flow_data)
pais = findPais(flow_data)

def find_id_pai():
    visit_data = pd.read_csv('Data/Echodata_KA_Trial.csv', sep=';')
    visit_data = visit_data[['roottable_case_id_text', 'clinical_visits_test_day_item','clinical_visits_test_day_date_date', 'PatientId']]
    pre_pai_id = {}
    mid_pai_id = {}
    post_pai_id = {}
    pre_id_pai = {}
    mid_id_pai = {}
    post_id_pai = {}
    #pre_days = []
    #mid_days = []
    #post_days = []
    for d in range(len(visit_data.clinical_visits_test_day_item.values)):
        pai = visit_data.PatientId.values[d]
        if visit_data.clinical_visits_test_day_item.values[d] == 'Pre-test day 2':
            pre_pai_id[pai] = int(visit_data.roottable_case_id_text.values[d])
            pre_id_pai[int(visit_data.roottable_case_id_text.values[d])] = pai
            #pre_days.append(pai)
        elif visit_data.clinical_visits_test_day_item.values[d] == 'Mid-test':
            mid_pai_id[pai] = int(visit_data.roottable_case_id_text.values[d])
            mid_id_pai[int(visit_data.roottable_case_id_text.values[d])] = pai
            #mid_days.append(pai)
        elif visit_data.clinical_visits_test_day_item.values[d] == 'Post-test day 2':
            post_pai_id[pai] = int(visit_data.roottable_case_id_text.values[d])
            post_id_pai[int(visit_data.roottable_case_id_text.values[d])] = pai
            #post_days.append(pai)
    
    return pre_id_pai, mid_id_pai, post_id_pai          #pre_days, mid_days, post_days

pre_id_pai, mid_id_pai, post_id_pai  = find_id_pai()

def structureFlowData(raw_flow, raw_time):
    remove_indicies = []
    for i in range(len(raw_flow)-1):
        if (raw_flow[i] == ',' and raw_flow[i+1] == ','):
            remove_indicies.append(i)
        elif (raw_flow[i] == '[' and raw_flow[i+1] == ','):
            remove_indicies.append(i+1)
    flow = raw_flow
    for i in reversed(remove_indicies):
        flow = flow[:i] + flow[i+1:]
    if not (flow[1] == 'n'):
        flow = literal_eval(flow)
    else: 
        flow = 0
    remove_indicies = []
    for i in range(len(raw_time)-1):
        if (raw_time[i] == ',' and raw_time[i+1] == ','):
            remove_indicies.append(i)
        elif (raw_time[i] == '[' and raw_time[i+1] == ','):
            remove_indicies.append(i+1)
    time = raw_time
    for i in reversed(remove_indicies):
        time = time[:i] + time[i+1:]
    if not (time[1] == 'n'):
        time = literal_eval(time)
    else: 
        time = 0
    return flow, time

def findFlow(patientId):
    pre_flow, pre_time, mid_time, mid_flow, post_time, post_flow = 0,0,0,0,0,0
    if patientId in pre_id_pai:
        pai = pre_id_pai[patientId]
        df = flow_data[(flow_data['patient'] == pai) &(flow_data['measurement'] == 'LVOT-VolumetricFlow-Manual')]
        if len(df.value.values) == 0:
            data = 0
            time = 0
        else:
            data = df.value.values[0]
            time = df.raw_time.values[0]
        if not data == 0:
            data = data.replace(' ',',')
            time = time.replace(' ',',')
            flow, time = structureFlowData(data, time)
            pre_flow = flow
            pre_time = time
    
    if patientId in mid_id_pai:
        pai = mid_id_pai[patientId]
        df = flow_data[(flow_data['patient'] == pai) &(flow_data['measurement'] == 'LVOT-VolumetricFlow-Manual')]
        if len(df.value.values) == 0:
            data = 0
            time = 0
        else:
            data = df.value.values[0]
            time = df.raw_time.values[0]
        if not data == 0:
            data = data.replace(' ',',')
            time = time.replace(' ',',')
            flow, time = structureFlowData(data, time)
            mid_flow = flow
            mid_time = time
 
    if patientId in post_id_pai:
        pai = post_id_pai[patientId]
        df = flow_data[(flow_data['patient'] == pai) &(flow_data['measurement'] == 'LVOT-VolumetricFlow-Manual')]
        if len(df.value.values) == 0:
            data = 0
            time = 0
        else:
            data = df.value.values[0]
            time = df.raw_time.values[0]
        if not data == 0:
            data = data.replace(' ',',')
            time = time.replace(' ',',')
            flow, time = structureFlowData(data, time)
            post_flow = flow
            post_time = time

    return pre_flow, pre_time, mid_flow, mid_time, post_flow, post_time

def scale_x(flow, time):
    flow_scaled = []
    start = time[0]
    end = time[-1]
    for n in range(0,100,1):
        f = np.interp(start+n/100*(end-start),time,flow)
        flow_scaled.append(f)

    return flow_scaled

def scale_curve(cycle, peak_location, end_systole_location, old_end_systole):
        scaled_time = list(range(0,100,1))
        end_diastole = 0
        peak = np.argmax(cycle)
        end = old_end_systole

        for i in range(0,peak,1):
            norm_dist_from_start = (i-0)/(peak-0)
            norm_dist_from_peak = 1-norm_dist_from_start
            new_x = peak_location * norm_dist_from_start + 0 * norm_dist_from_peak
            scaled_time[i] = new_x
        for i in range(peak,end+1,1):
            norm_dist_from_peak = (i-peak)/(end-peak)
            norm_dist_from_end = 1-norm_dist_from_peak
            new_x = peak_location * norm_dist_from_end + end_systole_location * norm_dist_from_peak
            scaled_time[i] = new_x
        return scaled_time
        
def standardizeFlow(flow, time):
    # Plot original flow curve
    '''
    plt.plot(time, flow, color='midnightblue')
    plt.ylabel('Volumetric Flow [m$^3$/s]')
    plt.xlabel('Time Points [-]')
    plt.show()
    '''
    largest_1 = 0
    largest_2 = 0
    split_1 = 0 
    split_2 = 0
    mean_flow = 0
    # find the periods between the three cycles
    if not (flow == 0):
        for t in range(len(time)-1):
            if (time[t+1]-time[t]) > largest_1:
                largest_2 = largest_1
                largest_1 = time[t+1]-time[t]
                split_2 = split_1
                split_1 = t+1
            elif (time[t+1]-time[t]) > largest_2:
                largest_2 = time[t+1]-time[t]
                split_2 = t+1
        # first and second split
        s1 = min(split_1, split_2)
        s2 = max(split_1, split_2)

        # first cycle 
        flow_1 = flow[:s1+1]
        time_1 = [x - time[:s1+1][0] for x in time[:s1+1]]
        # end systole location
        d1 = time_1[-2]/time_1[-1]
        d1 = int(d1*100)
        # second cycle
        flow_2 = flow[s1:s2+1]
        time_2 = [x - time[s1:s2+1][0] for x in time[s1:s2+1]]
        # end systole location
        d2 = time_2[-2]/time_2[-1]
        d2 = int(d2*100)
        # third cycle
        flow_3 = flow[s2:]
        flow_3.append(flow_1[-1])
        time_3 = [x - time[s2:][0] for x in time[s2:]]
        time_3.append(time_1[-1])
        # end systole location
        d3 = time_3[-2]/time_3[-1]
        d3 = int(d3*100)
        # scale cycles, x-axis 0-100
        flow_1_scaled = scale_x(flow_1,time_1)
        flow_2_scaled = scale_x(flow_2,time_2)
        flow_3_scaled = scale_x(flow_3,time_3)

        a = list(range(d1+1,len(flow_1_scaled)))
        b = list(range(d2+1,len(flow_2_scaled)))
        c = list(range(d3+1,len(flow_3_scaled)))
        for i in a:
            flow_1_scaled[i] = 0
        flow_1_scaled.insert(0,0)
        flow_1_scaled.pop()
        for i in b:
            flow_2_scaled[i] = 0
        flow_2_scaled.insert(0,0)
        flow_2_scaled.pop()
        for i in c:
            flow_3_scaled[i] = 0
        flow_3_scaled.insert(0,0)
        flow_3_scaled.pop()

        # mean end systole location
        es1 = d1+2
        es2 = d2+2
        es3 = d3+2
        es = list([es1, es2, es3])
        end_systole = int(round(sum(es)/float(len(es))))
        # mean systolic peak location and value 
        s1 = np.argmax(flow_1_scaled)
        s2 = np.argmax(flow_2_scaled)
        s3 = np.argmax(flow_3_scaled)
        systole_peak_i = list([s1,s2,s3])
        systole_peak_i = int(round(sum(systole_peak_i)/float(len(systole_peak_i))))
        p1 = max(flow_1_scaled)
        p2 = max(flow_2_scaled)
        p3 = max(flow_3_scaled)
        systole_peak = list([p1,p2,p3])
        systole_peak = int(round(sum(systole_peak)/float(len(systole_peak))))

        num = 0
        d = [d1,d2,d3]
        scaled_flows = []
        for c in [flow_1_scaled,flow_2_scaled,flow_3_scaled]:
            scaled_time = scale_curve(c,systole_peak_i,end_systole,d[num]+2)
            scaled_flow = scale_x(c, scaled_time)
            scaled_flows.append(scaled_flow)
            num += 1

        mean_flow = np.mean(scaled_flows, axis = 0)
        mean_time = np.mean(list([time_1[-1]-time_1[0],time_2[-1]-time_2[0],time_3[-1]-time_3[0]]))

    return mean_flow, mean_time
        
def allFlows(ids):
    pre_ids = []
    pre_flows = []
    scaled_pre_flows = []
    mid_ids = []
    mid_flows = []
    scaled_mid_flows = []
    post_ids = []
    post_flows = []
    scaled_post_flows = []
    pre_times =[]
    mid_times =[]
    post_times =[]
    for i in ids:
        pre_flow, pre_time, mid_flow, mid_time, post_flow, post_time = findFlow(i)
        pre_SV, mid_SV, post_SV = find_SV(i)
        if not (pre_flow == 0):
            pre_ids.append(i)
            flow, true_time = standardizeFlow(pre_flow,pre_time)
            pre_flows.append(flow)
            pre_times.append(true_time)
            integral = np.trapz(flow, np.linspace(0,true_time,100))
            scaled_flow = flow * (pre_SV/integral)
            scaled_pre_flows.append(scaled_flow)

            #savetxt('Data/Flow/Pre_flow_patient_' + str(i) + '.csv', flow, delimiter=',')

        if not (mid_flow == 0):
            mid_ids.append(i)
            flow, true_time = standardizeFlow(mid_flow,mid_time)
            mid_flows.append(flow)
            mid_times.append(true_time)
            integral = np.trapz(flow, np.linspace(0,true_time,100))
            scaled_flow = flow * (mid_SV/integral)
            scaled_mid_flows.append(scaled_flow)

            #savetxt('Data/Flow/Mid_flow_patient_' + str(i) + '.csv', flow, delimiter=',')

        if not (post_flow == 0):
            post_ids.append(i)
            flow, true_time = standardizeFlow(post_flow,post_time)
            post_flows.append(flow)
            post_times.append(true_time)
            integral = np.trapz(flow, np.linspace(0,true_time,100))
            scaled_flow = flow * (post_SV/integral)
            scaled_post_flows.append(scaled_flow)

            #savetxt('Data/Flow/Post_flow_patient_' + str(i) + '.csv', flow, delimiter=',')

    return pre_ids, mid_ids, post_ids, pre_flows, mid_flows, post_flows, pre_times, mid_times, post_times

pre_ids, mid_ids, post_ids, pre_flows, mid_flows, post_flows, pre_times, mid_times, post_times = allFlows(ids)
