import json
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import glob
import csv
from collections import Counter
from ast import literal_eval
import flow_data

def scale_x(flow, time):
    flow_scaled = []
    start = time[0]
    end = time[-1]
    for n in range(0,100,1):
        f = np.interp(start+n/100*(end-start),time,flow)
        flow_scaled.append(f)
    return flow_scaled

path = 'Data/Results_3wk_and_closedloop_030521/Results_all_participants/'
files = []
ids = []
for file in glob.glob(path+"PAI_*_trialId_*_ExerciseData.json"):
    with open(file) as jsonfile:
        reader = pd.read_json(jsonfile)
        files.append(reader)
        i = reader.trial_id.values[0]
        ids.append(i)

path_2 = 'Data/PreCurves_MM_and_3wk/PreCurves/'
pre_curves_dic = {}
for file in glob.glob(path_2+"PAI_*_trialId_*_PreCurves_and_Data.json"):
    with open(file) as jsonfile:
        reader = pd.read_json(jsonfile)
        pre = reader.P_ao.values[0]
        time = reader['T'].values[0]
        pre_curves_dic[reader.trial_id.values[0]] = (scale_x(pre, np.linspace(0,time,num=len(pre))))

pre_curves = []
for i in ids:
    pre_curves.append(pre_curves_dic[i])

curves = [] 
all_x = [] 
x_refs = [] 
true_exercise_estimates = [] 
periods = []
for f in files:
    true_exercise = f.loc[f['x'] == f['x_ref']]
    time = f['T'].values[0]
    true_exercise = scale_x(true_exercise.P_ao.values[0],np.linspace(0,time,num=len(true_exercise.P_ao.values[0])))
    true_exercise_estimates.append(true_exercise)
    periods.append(f['T'].values[0])
    t = []
    pat_curves = []
    for p in f.P_ao.values:
        c = scale_x(p,np.linspace(0,time,num=len(p)))
        pat_curves.append(c)
    for x in f.x.values:
        t.append(x)
    curves.append(pat_curves)
    all_x.append(t)
    x_refs.append(f.x_ref.values[0])

data_set = pd.read_csv('Data/full_data_set.csv')

exercise_dataset = data_set.copy()
empty_list = [[0]*100] * exercise_dataset.shape[0]
empty_x = [float(0)] * exercise_dataset.shape[0]
exercise_dataset['mm_post_finger_pressure_cycle'] = empty_list
exercise_dataset['mm_pre_finger_pressure_cycle'] = empty_list
exercise_dataset['exercise_value'] = empty_x
exercise_dataset['exercise_ref'] = empty_x
exercise_dataset['period'] = empty_list

# drop patients not estimated by MM - exercise dataset
for p in range(exercise_dataset.shape[0]):
    if exercise_dataset['roottable_case_id_text'][p] not in ids:
        exercise_dataset.drop(p,axis=0, inplace=True)
exercise_dataset.reset_index(drop=True, inplace=True)
simple_dataset = exercise_dataset.copy()

# add 150 curves per patient
for e in range(150):
    for i in range(len(ids)):
        pat = ids[i]
        t = all_x[i]
        for n in range(len(ids)):
            if exercise_dataset['roottable_case_id_text'][e*len(ids) + n] == pat:
                exercise_dataset.at[e*len(ids) + n, 'mm_pre_finger_pressure_cycle'] = pre_curves[i]
                exercise_dataset.at[e*len(ids) + n, 'mm_post_finger_pressure_cycle'] = curves[i][e]
                exercise_dataset.at[e*len(ids) + n, 'exercise_value'] = t[e]
                exercise_dataset.at[e*len(ids) + n, 'exercise_ref'] = x_refs[i]
                exercise_dataset.at[e*len(ids) + n, 'period'] = periods[i]
    if e == 149:
        break
    exercise_dataset = pd.concat([exercise_dataset,simple_dataset])
    exercise_dataset.reset_index(drop=True, inplace=True)

exercise_sorted = exercise_dataset.sort_values(by=['roottable_case_id_text', 'exercise_value'])
exercise_sorted.reset_index(drop=True, inplace=True)


# Plot mechanistic model estimates
'''
for p in range(23):
    print(exercise_sorted.clinical_visits_pre_24h_dbp_mean_value[p*150+i])
    print(exercise_sorted.clinical_visits_pre_24h_sbp_mean_value[p*150+i])
    for i in range(150):
        if i%10 == 0:
            plt.plot(exercise_sorted.mm_pre_finger_pressure_cycle.values[p*150+i], color='r')
            plt.plot(exercise_sorted.mm_post_finger_pressure_cycle.values[p*150+i], label=exercise_sorted.exercise_value.values[p*150+i], color='b')
    plt.legend()
    plt.show()
quit()
'''
# Plot mechanistic model estimates
'''
for i in range(0,150,45):
    plt.plot(exercise_dataset.sort_values(by=['roottable_case_id_text', 'exercise_value']).mm_post_finger_pressure_cycle.values[i], label=exercise_dataset.sort_values(by=['roottable_case_id_text', 'exercise_value']).exercise_value.values[i]) #, color="#2b2e4a")
for i in range(15,150,45):
    plt.plot(exercise_dataset.sort_values(by=['roottable_case_id_text', 'exercise_value']).mm_post_finger_pressure_cycle.values[i], label=exercise_dataset.sort_values(by=['roottable_case_id_text', 'exercise_value']).exercise_value.values[i]) #, color="#e84545")
for i in range(30,150,45):
    plt.plot(exercise_dataset.sort_values(by=['roottable_case_id_text', 'exercise_value']).mm_post_finger_pressure_cycle.values[i], label=exercise_dataset.sort_values(by=['roottable_case_id_text', 'exercise_value']).exercise_value.values[i]) #, color="#903749")
plt.legend()
plt.show()
quit()
'''


exercise_dataset = exercise_dataset[['roottable_age_value', 'roottable_sex_item','clinical_visits_body_mass_index_value','clinical_visits_cpet_vo2max_value',
                                'clinical_visits_pre_24h_dbp_mean_value','clinical_visits_pre_24h_sbp_mean_value', 'mm_pre_finger_pressure_cycle','period','exercise_ref','exercise_value','mm_post_finger_pressure_cycle']]    

exercise_dataset.to_csv(r'/Users/anineahlsand/iCloud Drive (arkiv)/Documents/Dokumenter/Documents/Skole/NTNU/Master/Code2021/Master2021/Data/synthetic_dataset.csv',index=False)

print('File saved')

