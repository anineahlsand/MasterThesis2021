import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from ast import literal_eval
import my_functions
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
np.set_printoptions(threshold=np.inf)

df = pd.read_csv('../../Master/Master2021/Data/full_data_set_double.csv', header=0)
df = df.sort_values(by=['roottable_case_id_text'])
df.reset_index(drop=True,inplace=True)

x_train = []
x_test = []
x_pred = []
pre_train = []
pre_test = []
pre_pred = []
post_train = []
post_test = []
post_pred = []
id_test = []
id_pred = []
ids_train = []

for index, row in df.iterrows():
    if sum(literal_eval(df.post_finger_pressure_cycle[index])) == 0:
        continue
    if pd.isna(row['clinical_visits_post_24h_dbp_mean_value']) or pd.isna(row['clinical_visits_post_24h_sbp_mean_value']):
        continue
    if pd.isna(row['clinical_visits_pre_24h_sbp_mean_value']) or pd.isna(row['clinical_visits_pre_24h_dbp_mean_value']):
        continue
    if df.roottable_case_id_text.values[index] == 59:
        x_test.append([df.roottable_age_value.values[index], df.roottable_sex_item.values[index], df.clinical_visits_body_mass_index_value.values[index], df.clinical_visits_cpet_vo2max_value.values[index],
                    df.clinical_visits_pre_24h_dbp_mean_value.values[index], df.clinical_visits_pre_24h_sbp_mean_value.values[index], df.MeanPAIPerDay.values[index]])
        pre_test.append(literal_eval(df.pre_finger_pressure_cycle[index]))
        post_test.append(literal_eval(df.post_finger_pressure_cycle[index]))
        id_test.append(df.roottable_case_id_text.values[index])
    if df.roottable_case_id_text.values[index] == 51:
        continue
    if df.roottable_case_id_text.values[index] == 93:
        continue
    if df.roottable_case_id_text.values[index] == 541:
        continue
    if df.roottable_case_id_text.values[index] == 578:
        continue
    else:
        x_train.append([df.roottable_age_value.values[index], df.roottable_sex_item.values[index], df.clinical_visits_body_mass_index_value.values[index], df.clinical_visits_cpet_vo2max_value.values[index],
                    df.clinical_visits_pre_24h_dbp_mean_value.values[index], df.clinical_visits_pre_24h_sbp_mean_value.values[index], df.MeanPAIPerDay.values[index]])
        pre_train.append(literal_eval(df.pre_finger_pressure_cycle[index]))
        post_train.append(literal_eval(df.post_finger_pressure_cycle[index]))
        ids_train.append(df.roottable_case_id_text.values[index])

x_train = np.array(x_train)
x_test = np.array(x_test)
post_train = np.array(post_train)
post_test = np.array(post_test)


min_max_scaler = MinMaxScaler()
x_train = min_max_scaler.fit_transform(x_train)
x_test = min_max_scaler.transform(x_test)

ts_train = []
ts_train.extend(pre_train)
ts_train.extend(post_train)
ts_train = np.array(ts_train)
mean_ts = my_functions.make_mean_vector(ts_train)
ts_wo_mean = my_functions.subtract_mean_from_post_ts_data(ts_train, mean_ts)


#### PCA ####
pca_model = PCA(0.95)
loadings = pca_model.fit_transform(ts_wo_mean)
components = pca_model.components_

## ENDRE TRAINX
x_train_transformed = []
for person in range(len(x_train)):
    for t in range(len(components[0])):
        rad = []
        for feat in range(len(x_train[person])):
            for c in range(len(components)):
                rad.append(x_train[person][feat] * components[c][t])
        x_train_transformed.append(rad)


### LINEAR REGRESSION ###
model_linear = LinearRegression()
model_linear.fit(x_train_transformed, post_train.flatten())
beta = model_linear.coef_
beta_0 = model_linear.intercept_


point_error = []
total_cycle_error = []
dbp_error = []
sbp_error = []
pp_error = []
MAP_error = []

##### NEW METHOD #####
predicted_ts = []
for t in range(len(components[0])):
    rad = []
    for feat in range(len(x_test[0])):
        for c in range(len(components)):
            rad.append(x_test[0][feat] * components[c][t])
    predicted_ts.append(np.dot(rad,beta))

predicted_ts_new = []
for point in range(len(predicted_ts)):
    predicted_ts_new.append((predicted_ts[point] + beta_0))


plt.plot(predicted_ts_new, color='darkorange', label = 'Prediction')
plt.plot(post_test[0], color='midnightblue', label= 'True curve')
plt.legend()
plt.title('Linear Regression')
plt.xlabel('Time points [-]')
plt.ylabel('Blood Pressure [mmHg]')
plt.show()


# Calculate erorrs
dbp = min(predicted_ts_new)
sbp = max(predicted_ts_new)
pp = sbp-dbp
MAP = np.mean(predicted_ts_new)
dbp_true = min(post_test[0])
sbp_true = max(post_test[0])
pp_true = sbp_true-dbp_true
MAP_true = np.mean(post_test[0])

point_error.append(abs(predicted_ts_new - post_test[0]))
total_cycle_error.append(sum(abs(predicted_ts_new - post_test[0])))
dbp_error.append(abs(dbp-dbp_true))
sbp_error.append(abs(sbp-sbp_true))
pp_error.append(abs(pp-pp_true))
MAP_error.append(abs(MAP-MAP_true))


print('Point error = ', np.mean(point_error), np.std(point_error))
print('DBP error = ', np.mean(dbp_error), np.std(dbp_error))
print('SBP error = ', np.mean(sbp_error), np.std(sbp_error))
print('PP error = ', np.mean(pp_error), np.std(pp_error))
print('MAP error = ', np.mean(MAP_error), np.std(MAP_error))
print('Total error = ', np.mean(total_cycle_error), np.std(total_cycle_error))

