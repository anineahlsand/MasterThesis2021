import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.utils import shuffle  
from itertools import chain
import tensorflow as tf
from tensorflow import keras
from keras import layers
import random
from ast import literal_eval
import my_functions
import random


def get_data():
    df = pd.read_csv('Data/synthetic_dataset.csv', header=0)
    full = pd.read_csv('Data/full_data_set.csv', header=0)
    return df

def split_data(df):
    training_set = df[['roottable_age_value', 'roottable_sex_item','clinical_visits_body_mass_index_value','clinical_visits_cpet_vo2max_value','clinical_visits_pre_24h_dbp_mean_value','clinical_visits_pre_24h_sbp_mean_value', 'exercise_value']]                              
    labels = df[['mm_post_finger_pressure_cycle']]

    return training_set, labels

def person_to_predict(df, training_set, labels):
    age = training_set.iloc[10,0]
    gender = training_set.iloc[10,1]
    bmi = training_set.iloc[10,2]
    vo2 = training_set.iloc[10,3]
    dbp = training_set.iloc[10,4]
    sbp = training_set.iloc[10,5]
    one_person = (df.loc[(df['roottable_age_value'] == age) & (df['roottable_sex_item'] == gender) & (df['clinical_visits_body_mass_index_value'] == bmi) & 
                    (df['clinical_visits_cpet_vo2max_value'] == vo2) & (df['clinical_visits_pre_24h_dbp_mean_value'] == dbp) & (df['clinical_visits_pre_24h_sbp_mean_value'] == sbp)]).sort_values(by=['exercise_value'])
    one_person = one_person[['roottable_age_value', 'roottable_sex_item','clinical_visits_body_mass_index_value','clinical_visits_cpet_vo2max_value',
                                'clinical_visits_pre_24h_dbp_mean_value','clinical_visits_pre_24h_sbp_mean_value' ,'exercise_value','mm_post_finger_pressure_cycle']]
    age_2 = training_set.iloc[6,0]
    gender_2 = training_set.iloc[6,1]
    bmi_2 = training_set.iloc[6,2]
    vo2_2 = training_set.iloc[6,3]
    dbp_2 = training_set.iloc[6,4]
    sbp_2 = training_set.iloc[6,5]
    second_person = (df.loc[(df['roottable_age_value'] == age_2) & (df['roottable_sex_item'] == gender_2) & (df['clinical_visits_body_mass_index_value'] == bmi_2) & 
                    (df['clinical_visits_cpet_vo2max_value'] == vo2_2) & (df['clinical_visits_pre_24h_dbp_mean_value'] == dbp_2) & (df['clinical_visits_pre_24h_sbp_mean_value'] == sbp_2)]).sort_values(by=['exercise_value'])
    second_person = second_person[['roottable_age_value', 'roottable_sex_item','clinical_visits_body_mass_index_value','clinical_visits_cpet_vo2max_value',
                                'clinical_visits_pre_24h_dbp_mean_value','clinical_visits_pre_24h_sbp_mean_value' ,'exercise_value','mm_post_finger_pressure_cycle']]

    test_pers = pd.DataFrame().reindex_like(one_person)
    pred_pers = pd.DataFrame().reindex_like(second_person)
    drop_ind = []
    a = 0
    for i in list(one_person.index):
        drop_ind.append(i)
        test_pers.loc[a] = one_person.iloc[a,:]
        a += 1
    test_pers = test_pers.dropna()
    test_pers.reset_index(drop=True, inplace=True)
    x_test_pers = test_pers.iloc[:,:-1]
    y_test_pers = test_pers.iloc[:,-1]
    x_test_pers = np.array(x_test_pers)
    y_test_pers = np.array(y_test_pers)
    a = 0
    for i in list(second_person.index):
        drop_ind.append(i)
        pred_pers.loc[a] = second_person.iloc[a,:]
        a += 1
    pred_pers = pred_pers.dropna()
    pred_pers.reset_index(drop=True, inplace=True)
    x_pred_pers = pred_pers.iloc[:,:-1]
    y_pred_pers = pred_pers.iloc[:,-1]
    x_pred_pers = np.array(x_pred_pers)
    y_pred_pers = np.array(y_pred_pers)

    new_training_set = training_set.drop(drop_ind, axis=0)
    new_labels = labels.drop(drop_ind)

    new_training_set = np.array(new_training_set)
    new_labels = np.array(new_labels)

    return x_test_pers, y_test_pers, x_pred_pers, y_pred_pers, new_training_set, new_labels

def split_100(x,y):
    x_new = []
    y_new = []
    for p in range(len(x)):
        t = 0.00
        for i in range(100):
            x_new.append(list(chain(x[p], [t])))
            y_new.append(y[p][i])
            t += 0.01
    return x_new, y_new

def get_x_y(training_set, labels, x_test_pers, y_test_pers, x_pred_pers, y_pred_pers):
    x = training_set
    min_max_scaler = MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(np.concatenate((x,x_test_pers,x_pred_pers),axis=0))
    x_test = x_scaled[-300:-150][:]
    x_pred = x_scaled[-150:][:]
    x_norm = x_scaled.tolist()
    for i in range(len(x_test)*2):
        x_norm.pop(-1)
    x_norm = np.reshape(x_norm, (len(x_norm),len(x_norm[0])))

    y_test = []
    for s in y_test_pers:
        y_test.append(literal_eval(s))

    y_pred = []
    for s in y_pred_pers:
        y_pred.append(literal_eval(s))

    y_post = []
    for y in range(len(labels)):
        raw_curve_post = labels[y][0]
        y_post.append(literal_eval(raw_curve_post))

    # Choose share of dataset for training
    x_norm, y_post = zip(*random.sample(list(zip(list(x_norm), y_post)), int(x_norm.shape[0]*0.6)))
    x_norm = np.array(x_norm).reshape(len(x_norm),len(x_norm[0]))
    
    print('Train shape: ', x_norm.shape)
    # Split into train and test
    x_train, x_val, y_train, y_val = train_test_split(x_norm, y_post, test_size=0.1, shuffle= True)

    x_train = np.array(x_train)
    x_val = np.array(x_val)
    x_test = np.array(x_test)
    x_pred = np.array(x_pred)
    y_train = np.array(y_train)
    y_val = np.array(y_val)
    y_test = np.array(y_test)
    y_pred = np.array(y_pred)
   
    return x_train, x_val, x_test, x_pred, y_train, y_val, y_test, y_pred

def create_model():
    inputs = keras.Input(shape=(None,7))
    x1 = keras.layers.Dense(25,activation="relu")(inputs)
    x2 = keras.layers.Dense(75,activation="relu")(x1)
    outputs = keras.layers.Dense(100, activation="linear")(x2)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
                loss='mean_squared_error',
                optimizer=keras.optimizers.Adam(lr=0.01),
                metrics=["mean_squared_error"]
                )
    return model

def fit_model(model, x_train, x_test, y_train, y_test):

    history = model.fit(x_train, y_train, epochs=100, batch_size=32,validation_data=(x_test, y_test), verbose=0)
    model.summary()

    # Evaluate the model
    train_mse = model.evaluate(x_train, y_train, verbose=0)
    test_mse = model.evaluate(x_test, y_test, verbose=0)
    print('Train loss: %.3f, Test: %.3f' % (train_mse[0], test_mse[0]))
    '''
    # Plot loss during training
    plt.title('100% of total dataset ')
    plt.plot(history.history['loss'], label='Train', color='#53354a')
    plt.plot(history.history['val_loss'], label='Test', color='#e84545')
    plt.ylabel('Loss / MSE')
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()
    '''
    return model

def predict(x_predict, y_predict):
    predictions = []
    total_cycle_error = []
    total_error_short = []
    dbp_error = []
    sbp_error = []
    pp_error = []
    MAP_error = []
    point_error = []
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for p in range(150):
        pers = x_predict[p].reshape(1,-1)
        prediction = model.predict(pers)[0]
        true = y_predict[p]

        # Calculate erorrs 
        predictions.append(prediction)
        dbp = min(prediction)
        sbp = max(prediction)
        pp = sbp-dbp
        MAP = np.mean(prediction)
        dbp_true = min(true)
        sbp_true = max(true)
        pp_true = sbp_true-dbp_true
        MAP_true = np.mean(true)
        point_error.append(abs(prediction-true))
        total_cycle_error.append(sum(abs(prediction-true)))
        total_error_short.append(sum(abs(prediction[5:90]-true[5:90]))/85*100)
        dbp_error.append(abs(dbp-dbp_true))
        sbp_error.append(abs(sbp-sbp_true))
        pp_error.append(abs(pp-pp_true))
        MAP_error.append(abs(MAP-MAP_true))
        if sbp > 140 or dbp > 90:
            pred_pos = True 
        else:
            pred_pos = False
        if sbp_true > 140 or dbp_true > 90:
            pos = True
        else:
            pos = False

        if pred_pos == True:
            if pos == True:
                TP += 1
            else:
                FP += 1
        else:
            if pos == True:
                FN += 1
            else:
                TN += 1
    
    point_error = [item for sublist in point_error for item in sublist]
    
    print('DBP error = ', np.mean(dbp_error), np.std(dbp_error))
    print('SBP error = ', np.mean(sbp_error), np.std(sbp_error))
    print('PP error = ', np.mean(pp_error), np.std(pp_error))
    print('MAP error = ', np.mean(MAP_error), np.std(MAP_error))
    print('Point error = ', np.mean(point_error), np.std(point_error))
    print('True positives = ', TP)
    print('False positives = ', FP)
    print('True negatives = ', TN)
    print('False negatives = ', FN)
    print('\n')
    
    return np.mean(total_cycle_error)


df = get_data()
training_set, labels = split_data(df)
x_test_pers, y_test_pers, x_pred_pers, y_pred_pers, training_set, labels = person_to_predict(df, training_set, labels)
x_train, x_val, x_test, x_pred, y_train, y_val, y_test, y_pred = get_x_y(training_set, labels, x_test_pers, y_test_pers, x_pred_pers, y_pred_pers)
'''
best_error = np.inf
for i in range(100):
    model = create_model()
    model = fit_model(model,x_train,x_val,y_train,y_val)
    error = predict(x_test, y_test)
    #if error < this_best:
    if error < best_error:
        print('\n New best error on number ',i,'. Error = ', error, '\n')
        model.save_weights('Data/synth_weights_10_1')
        best_error = error
print('The best error is: ', best_error)

quit()
'''
model = create_model()
model.load_weights('Data/synth_weights_10_60')
error = predict(x_test, y_test)
#print('The error of prediction is: ', error)

    
