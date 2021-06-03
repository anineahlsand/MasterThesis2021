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

curves = []
all_x = []
x_refs = []
time = []
test_ids = []
true_exercise_estimates = []
for f in files:
    time.append(round(f['T'][0], 3))
    true_exercise = f.loc[f['x'] == f['x_ref']]
    #print(true_exercise)
    # true_exercise = true_exercise.P_ao.values[0]
    # te = true_exercise.trial_id.values[0]
    # test_ids.append(te)
    true_exercise = scale_x(true_exercise.P_ao.values[0],range(0,200))
    true_exercise_estimates.append(true_exercise)
    t = []
    pat_curves = []
    for p in f.P_ao.values:
        c = scale_x(p,range(0,200))
        pat_curves.append(c)
    for x in f.x.values:
        t.append(x)
    curves.append(pat_curves)
    all_x.append(t)
    x_refs.append(f.x_ref.values[0])

data_set = pd.read_csv('Data/full_data_set.csv')

# print(ids)
# # print(test_ids)
# print(data_set.shape)
# quit()

residual_dataset = data_set.copy()
empty_list = [[0]*100] * residual_dataset.shape[0]
empty_x = [float(0)] * residual_dataset.shape[0]

# print(residual_dataset.shape)
# print(residual_dataset.columns)

residual_dataset['mm_post_finger_pressure_cycle'] = empty_list
residual_dataset['exercise_value'] = empty_x
residual_dataset['estimate_error'] = empty_list
residual_dataset['true_post_finger_pressure_cycle'] = empty_list
residual_dataset['cycle_time'] = empty_x

# drop patients not estimated by MM
for p in range(residual_dataset.shape[0]):
    if residual_dataset['roottable_case_id_text'][p] not in ids:
        residual_dataset.drop(p,axis=0, inplace=True)
residual_dataset.reset_index(drop=True, inplace=True)


for i in range(len(ids)):
    pat = ids[i]
    for n in range(residual_dataset.shape[0]):
        if residual_dataset['roottable_case_id_text'][n] == pat:
            post_cycle = residual_dataset.loc[n,'post_finger_pressure_cycle']
            post_cycle = literal_eval(post_cycle)
            residual_dataset.at[n,'true_post_finger_pressure_cycle'] = post_cycle
            residual_dataset.at[n,'cycle_time'] = time[i]
            residual_dataset.at[n, 'mm_post_finger_pressure_cycle'] = true_exercise_estimates[i]
            residual_dataset.at[n, 'exercise_value'] = x_refs[i]

# for i in range(len(ids)):
#     pat = ids[i]
#     for n in range(residual_dataset.shape[0]):
#         if residual_dataset['roottable_case_id_text'][n] == pat:
#             residual_dataset.at[n, 'mm_post_finger_pressure_cycle'] = true_exercise_estimates[i]
#             residual_dataset.at[n, 'exercise_value'] = x_refs[i]


for i in range(residual_dataset.shape[0]):
    if not (residual_dataset.loc[i,'true_post_finger_pressure_cycle'][0] == 0):
        residual_dataset.at[i, 'estimate_error'] = [x1 - x2 for (x1, x2) in zip(residual_dataset.mm_post_finger_pressure_cycle.values[i], residual_dataset.true_post_finger_pressure_cycle.values[i])]

# print(residual_dataset[['roottable_case_id_text', 'mm_post_finger_pressure_cycle', 'true_post_finger_pressure_cycle', 'exercise_value']])
# quit()

residual_dataset = residual_dataset[['roottable_case_id_text', 'roottable_age_value', 'roottable_sex_item','clinical_visits_body_mass_index_value','clinical_visits_cpet_vo2max_value',
                                    'clinical_visits_pre_24h_dbp_mean_value','clinical_visits_pre_24h_sbp_mean_value','pre_finger_pressure_cycle', 
                                    'exercise_value','mm_post_finger_pressure_cycle','true_post_finger_pressure_cycle', 'estimate_error', 'cycle_time']]
'''
for i in range(residual_dataset.shape[0]):
    if not (residual_dataset.true_post_finger_pressure_cycle.values[i][0]) == 0:
        plt.plot(residual_dataset.estimate_error.values[i])
        plt.show()
quit()
'''
#exercise_dataset.to_csv(r'/Users/anineahlsand/iCloud Drive (arkiv)/Documents/Dokumenter/Documents/Skole/NTNU/Master/Code2021/Master2021/Data/generated_dataset.csv',index=False)
residual_dataset.to_csv(r'/Users/kariannedalheim/Documents/Master/Master2021/Data/residual_dataset_new.csv',index=False)
print(residual_dataset)
print('File saved')

