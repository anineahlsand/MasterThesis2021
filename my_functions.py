import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from ast import literal_eval

def preprocess_df(df):
    pre_ts = []
    post_ts = []
    input_var = []
    for index, row in df.iterrows():
        input_variables = []
        if pd.isna(row['clinical_visits_post_24h_dbp_mean_value']) or pd.isna(row['clinical_visits_post_24h_sbp_mean_value']):
            continue
        if pd.isna(row['clinical_visits_pre_24h_sbp_mean_value']) or pd.isna(row['clinical_visits_pre_24h_dbp_mean_value']):
            continue
        if sum(literal_eval(df.post_finger_pressure_cycle[index])) == 0:
            continue
        else:
            pre_ts.append(literal_eval(df.pre_finger_pressure_cycle[index]))
            post_ts.append(literal_eval(df.post_finger_pressure_cycle[index]))
            
            age = float(df.roottable_age_value[index])
            gender = float(df.roottable_sex_item[index])
            PAI = float(df.MeanPAIPerDay[index])
            vo2 = float(df.clinical_visits_cpet_vo2max_value[index])
            pre_dbp = float(df.clinical_visits_pre_24h_dbp_mean_value[index])
            pre_sbp = float(df.clinical_visits_pre_24h_sbp_mean_value[index])

            input_variables.append(age)
            input_variables.append(gender)
            input_variables.append(PAI)
            input_variables.append(vo2)
            input_variables.append(pre_dbp)
            input_variables.append(pre_sbp)
            input_variables.append((PAI*vo2))

            input_var.append(input_variables)
    
    pre_ts = np.array(pre_ts)
    post_ts = np.array(post_ts)

    return pre_ts, post_ts, input_var

def apply_pca(data, variance):
    X_all = data
    pca = PCA(variance)
    X_pca = pca.fit_transform(X_all)
    comp = pca.components_
    return comp, pca, X_pca

def normalize(new_person, all_data):
    normalized = np.zeros(len(new_person))
    for i in range(len(new_person)):
        global maxim
        maxim = 0
        global minim
        minim = np.inf
        for pers in all_data:
            if pers[i] > maxim:
                maxim = pers[i]
            if pers[i] < minim:
                minim = pers[i]
        normalized[i] = (new_person[i]-minim)/(maxim-minim)
    return normalized

def normalize_data_set(full_dataset, n_train): 
    input_vector_normalized = []
    for kolonne in full_dataset: 
        norm = normalize(kolonne, full_dataset)
        input_vector_normalized.append(norm)

    input_vector_normalized = np.array(input_vector_normalized)
    trainX_norm = input_vector_normalized[0:n_train]
    testX_norm = input_vector_normalized[n_train:]

    return input_vector_normalized, trainX_norm, testX_norm

def make_train_test_data(n_train, data_set):
    trainX = []
    testX = []
    person_data = []

    for s in range(n_train):
        trainX.append(data_set[:n_train][s][0:6])
        person_data.append(data_set[:n_train][s][0:6])
    for j in range(len(data_set) - n_train):
        testX.append(data_set[n_train:][j][0:6])
        person_data.append(data_set[n_train:][j][0:6])
    trainX = np.array(trainX)
    testX = np.array(testX)
    person_data = np.array(person_data)

    return trainX, testX, person_data

def make_mean_vector(post_ts):
    mean_vector = []
    for t in range(len(post_ts[0])):
        col = []
        for o in range(len(post_ts)):
            col.append(post_ts[o][t])
        mean_vector.append(np.mean(col))
    return mean_vector

def subtract_mean_from_post_ts_data(post_ts, mean_vector):
    new_ts = post_ts.copy()
    for elem in range(len(new_ts[0])):
        for row in range(len(new_ts)):
            new_ts[row][elem] = post_ts[row][elem] - mean_vector[elem]
    return new_ts

def apply_linear_regression(trainX, trainy):
    X, y = trainX, trainy
    model = LinearRegression()
    model.fit(X, y)
    beta = model.coef_

    return model, beta

def apply_new_method(beta, component, test_pers):
    new_ts = []
    for t in range(len(component[0])):
        rad = []
        for feat in range(len(test_pers)):
            for c in range(len(component)):
                rad.append(test_pers[feat] * component[c][t])
        new_ts.append(np.dot(rad,beta))

    return new_ts

def add_mean_to_new_prediction(ts, mean_vector):
    for element in range(len(ts)):
        ts[element] = ts[element] + mean_vector[element]
    return ts

def add_mean_to_post_ts(time_series, mean_vector):
    for elem in range(len(time_series[0])):
        for col in range(len(time_series)):
            time_series[col][elem] = time_series[col][elem] + mean_vector[elem]
    return time_series