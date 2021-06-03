import json
from numpy.lib.shape_base import _put_along_axis_dispatcher
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import glob
import csv
from collections import Counter
from ast import literal_eval

from six import text_type
import flow_data
np.set_printoptions(threshold=np.inf)

def scale_x(flow, time):
    flow_scaled = []
    start = time[0]
    end = time[-1]
    for n in range(0,100,1):
        f = np.interp(start+n/100*(end-start),time,flow)
        flow_scaled.append(f)
    return flow_scaled

path = 'Data/Results_3wk_and_closedloop_030521/Results_all_participants/'
path1 = 'Data/Results_3wk_and_closedloop_030521/Results_3wk_postflow/'
path2 = 'Data/PreCurves_MM_and_3wk/PreCurves/'
path3 = 'Data/Results_3wk_modelflow/'
files = []
ids = []
test_pre = []
for file in glob.glob(path+"PAI_*_trialId_*_ExerciseData.json"):
    with open(file) as jsonfile:
        reader = pd.read_json(jsonfile)
        files.append(reader)
        i = reader.trial_id.values[0]
        ids.append(i)

# GAMLE KURVER: path1 - NYE KURVER: path3
ids1 = []
files1 = []
for file in glob.glob(path1+"PAI_*_trialId_*_ExerciseData.json"):
    with open(file) as jsonfile:
        reader = pd.read_json(jsonfile)
        files1.append(reader)
        i = reader.trial_id.values[0]
        ids1.append(i)
        test_pre.append(reader.P_ao.values[0])

# pre = []
# for file in glob.glob(path3+"PAI_*_trialId_*_ExerciseData.json"):
#     with open(file) as jsonfile:
#         reader = pd.read_json(jsonfile)
#         pressure = reader.P_ao.values[0]
#         print(len(pressure))
#         # pre.append(scale_x(pressure, range(0,260)))
#         pre.append(pressure)

pre_curves_dic = {}
id_test_rekkef = []
for file in glob.glob(path2+"PAI_*_trialId_*_PreCurves_and_Data.json"):
    with open(file) as jsonfile:
        reader = pd.read_json(jsonfile)
        i = reader.P_ao.values[0]
        pre_curves_dic[reader.trial_id.values[0]] = (scale_x(i, range(0,200)))
        id_test_rekkef.append(reader.trial_id.values[0])

pre_curves = []
for i in ids:
    pre_curves.append(pre_curves_dic[i])

# for k in range(23):
#     plt.plot(pre[k], 'r')
#     plt.plot(test_pre[k], 'pink')
# plt.show()
# quit()

mm_curves = []
all_x = []
x_refs = []
true_exercise_estimates = []
time = []
for f in files:
    time.append(round(f['T'][0], 3))
    true_exercise = f.loc[f['x'] == f['x_ref']]
    true_exercise = scale_x(true_exercise.P_ao.values[0],range(0,200))
    true_exercise_estimates.append(true_exercise)
    t = []
    pat_curves = []
    for p in f.P_ao.values:
        c = scale_x(p,range(0,200))
        pat_curves.append(c)
    for x in f.x.values:
        t.append(x)
    mm_curves.append(pat_curves)
    all_x.append(t)
    x_refs.append(f.x_ref.values[0])


wk3_curves = []
all_x_wk3 = []
x_refs_wk3 = []
for fil in files1:
    t = []
    pat_curves = []
    for p in fil.P_ao.values:
        c = scale_x(p,range(0,len(p)))
        pat_curves.append(c)
    for x in fil.x.values:
        t.append(x)
    wk3_curves.append(pat_curves)
    all_x_wk3.append(t)
    x_refs_wk3.append(fil.x_ref.values[0])


# for aynene in range(len(mm_curves)):
#     for kary in range(len(mm_curves[0])):
#         plt.plot(mm_curves[aynene][kary])
#         plt.plot(wk3_curves[aynene][kary])
#         plt.show()

data_set = pd.read_csv('Data/full_data_set.csv')

exercise_dataset = data_set.copy()
residual_dataset = data_set.copy()
exercise_residual_dataset = data_set.copy()
empty_list = [[0]*100] * exercise_dataset.shape[0]
empty_x = [float(0)] * exercise_dataset.shape[0]
exercise_dataset['mm_post_finger_pressure_cycle'] = empty_list
exercise_dataset['exercise_value'] = empty_x
exercise_dataset['exercise_ref'] = empty_x
residual_dataset['mm_post_finger_pressure_cycle'] = empty_list
residual_dataset['exercise_value'] = empty_x
residual_dataset['estimate_error'] = empty_list
residual_dataset['true_post_finger_pressure_cycle'] = empty_list
exercise_residual_dataset['mm_post_finger_pressure_cycle'] = empty_list
exercise_residual_dataset['mm_pre_finger_pressure_cycle'] = empty_list
exercise_residual_dataset['exercise_value'] = empty_x
exercise_residual_dataset['estimate_error'] = empty_list
exercise_residual_dataset['wk3_post_finger_pressure_cycle'] = empty_list
exercise_residual_dataset['cycle_time'] = empty_x
# exercise_residual_dataset['true_post_finger_pressure_cycle'] = empty_list
# exercise_residual_dataset['true_pre_finger_pressure_cycle'] = empty_list


# drop patients not estimated by MM
for p in range(exercise_residual_dataset.shape[0]):
    if exercise_residual_dataset['roottable_case_id_text'][p] not in ids:
        exercise_residual_dataset.drop(p,axis=0, inplace=True)
exercise_residual_dataset.reset_index(drop=True, inplace=True)
simple_dataset = exercise_residual_dataset.copy()


# add 150 curves per patient
for e in range(150):
    for i in range(len(ids)):
        pat = ids[i]
        t = all_x[i]
        cycle_time = time[i]
        for n in range(len(ids)):
            if exercise_residual_dataset['roottable_case_id_text'][e*len(ids) + n] == pat:
                exercise_residual_dataset.at[e*len(ids) + n, 'mm_post_finger_pressure_cycle'] = mm_curves[i][e]
                exercise_residual_dataset.at[e*len(ids) + n, 'mm_pre_finger_pressure_cycle'] = pre_curves[i]
                exercise_residual_dataset.at[e*len(ids) + n, 'wk3_post_finger_pressure_cycle'] = wk3_curves[i][e]
                exercise_residual_dataset.at[e*len(ids) + n, 'exercise_value'] = t[e]
                exercise_residual_dataset.at[e*len(ids) + n,'cycle_time'] = cycle_time
                exercise_residual_dataset.at[e*len(ids) + n, 'estimate_error'] = [x2 - x1 for (x1, x2) in zip(wk3_curves[i][e], mm_curves[i][e])]
    if e == 149:
        break
    exercise_residual_dataset = pd.concat([exercise_residual_dataset,simple_dataset])
    exercise_residual_dataset.reset_index(drop=True, inplace=True)

print(exercise_residual_dataset.shape)

print(exercise_residual_dataset)
# print(exercise_residual_dataset.3wk_post_finger_pressure_cycle)
# print(len(exercise_residual_dataset.3wk_post_finger_pressure_cycle))

'''
print(exercise_dataset.columns)
for i in range(0,150,45):
    plt.plot(exercise_dataset.sort_values(by=['roottable_case_id_text', 'exercise_value']).mm_post_finger_pressure_cycle.values[i], label=exercise_dataset.sort_values(by=['roottable_case_id_text', 'exercise_value']).exercise_value.values[i], color="#2b2e4a")
    plt.show()
    quit()
for i in range(15,150,45):
    plt.plot(exercise_dataset.sort_values(by=['roottable_case_id_text', 'exercise_value']).mm_post_finger_pressure_cycle.values[i], label=exercise_dataset.sort_values(by=['roottable_case_id_text', 'exercise_value']).exercise_value.values[i], color="#e84545")
for i in range(30,150,45):
    plt.plot(exercise_dataset.sort_values(by=['roottable_case_id_text', 'exercise_value']).mm_post_finger_pressure_cycle.values[i], label=exercise_dataset.sort_values(by=['roottable_case_id_text', 'exercise_value']).exercise_value.values[i], color="#903749")
plt.show()
quit()
'''
exercise_dataset = exercise_dataset[['roottable_age_value', 'roottable_sex_item','clinical_visits_body_mass_index_value','clinical_visits_cpet_vo2max_value',
                                'clinical_visits_pre_24h_dbp_mean_value','clinical_visits_pre_24h_sbp_mean_value', 'pre_finger_pressure_cycle','exercise_ref','exercise_value','mm_post_finger_pressure_cycle']]    
training_set = exercise_dataset[['roottable_age_value', 'roottable_sex_item','clinical_visits_body_mass_index_value','clinical_visits_cpet_vo2max_value',
                                'clinical_visits_pre_24h_dbp_mean_value','clinical_visits_pre_24h_sbp_mean_value', 'pre_finger_pressure_cycle','exercise_value']]
labels = exercise_dataset[['mm_post_finger_pressure_cycle']]
residual_dataset = residual_dataset[['roottable_age_value', 'roottable_sex_item','clinical_visits_body_mass_index_value','clinical_visits_cpet_vo2max_value',
                                    'clinical_visits_pre_24h_dbp_mean_value','clinical_visits_pre_24h_sbp_mean_value','pre_finger_pressure_cycle', 
                                    'exercise_value','mm_post_finger_pressure_cycle','true_post_finger_pressure_cycle', 'estimate_error']]
exercise_residual_dataset = exercise_residual_dataset[['roottable_case_id_text', 'roottable_age_value', 'roottable_sex_item', 'clinical_visits_body_mass_index_value', 
        'clinical_visits_cpet_vo2max_value','clinical_visits_pre_24h_dbp_mean_value','clinical_visits_pre_24h_sbp_mean_value', 'pre_finger_pressure_cycle', 'post_finger_pressure_cycle', 'mm_pre_finger_pressure_cycle', 'mm_post_finger_pressure_cycle', 'exercise_value', 'estimate_error',
       'wk3_post_finger_pressure_cycle', 'cycle_time']]

# for i in range(exercise_residual_dataset.shape[0]):
#     fig, axs = plt.subplots(2)
#     axs[0].plot(exercise_residual_dataset.mm_post_finger_pressure_cycle[i])
#     axs[0].plot(exercise_residual_dataset.wk3_post_finger_pressure_cycle[i])
#     axs[1].plot(exercise_residual_dataset.estimate_error.values[i])
#     plt.show()
# quit()

#exercise_dataset.to_csv(r'/Users/anineahlsand/iCloud Drive (arkiv)/Documents/Dokumenter/Documents/Skole/NTNU/Master/Code2021/Master2021/Data/generated_dataset.csv',index=False)
exercise_residual_dataset.to_csv(r'/Users/kariannedalheim/Documents/Master/Master2021/Data/exercise_residual_dataset_old.csv',index=False)
print(exercise_residual_dataset)
print('File saved')

