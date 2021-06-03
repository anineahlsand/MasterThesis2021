import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.models import Sequential
from tensorflow import keras
from tensorflow.keras import layers
from keras.optimizers import SGD, Adam
from keras.constraints import maxnorm
from keras.layers import Dense, Conv2D, MaxPooling2D
from ast import literal_eval
import my_functions
from operator import add
from itertools import chain
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
np.set_printoptions(threshold=np.inf)
import random

# Model configuration
batch_size = 30
n_train = 2350
loss_function = 'mean_squared_error'
no_epochs = 100
optimizer = Adam(lr=0.001)

def get_data():
    df = pd.read_csv('Data/exercise_residual_dataset.csv', header=0)
    df = df.sort_values(by=['roottable_case_id_text'])
    df.reset_index(drop=True,inplace=True)

    x_train = []
    x_test = []
    x_pred = []
    y_train = []
    y_test = []
    y_pred = []
    MM_train = []
    MM_test = []
    MM_pred = []
    wk_train = []
    wk_test = []
    wk_pred = []
    id_test = []
    id_pred = []

    for index, row in df.iterrows():
        if df.roottable_case_id_text.values[index] == 17:
            x_pred.append([df.roottable_age_value.values[index], df.roottable_sex_item.values[index], df.clinical_visits_body_mass_index_value.values[index], df.clinical_visits_cpet_vo2max_value.values[index],
                        df.clinical_visits_pre_24h_dbp_mean_value.values[index], df.clinical_visits_pre_24h_sbp_mean_value.values[index], df.exercise_value.values[index]])
            y_pred.append(literal_eval(df.estimate_error[index]))
            MM_pred.append(literal_eval(df.mm_post_finger_pressure_cycle[index]))
            wk_pred.append(literal_eval(df.wk3_post_finger_pressure_cycle[index]))
            id_pred.append(df.roottable_case_id_text.values[index])
        elif df.roottable_case_id_text.values[index] == 217:
            x_test.append([df.roottable_age_value.values[index], df.roottable_sex_item.values[index], df.clinical_visits_body_mass_index_value.values[index], df.clinical_visits_cpet_vo2max_value.values[index],
                        df.clinical_visits_pre_24h_dbp_mean_value.values[index], df.clinical_visits_pre_24h_sbp_mean_value.values[index], df.exercise_value.values[index]])
            y_test.append(literal_eval(df.estimate_error[index]))
            MM_test.append(literal_eval(df.mm_post_finger_pressure_cycle[index]))
            wk_test.append(literal_eval(df.wk3_post_finger_pressure_cycle[index]))
            id_test.append(df.roottable_case_id_text.values[index])
        else:
            x_train.append([df.roottable_age_value.values[index], df.roottable_sex_item.values[index], df.clinical_visits_body_mass_index_value.values[index], df.clinical_visits_cpet_vo2max_value.values[index],
                        df.clinical_visits_pre_24h_dbp_mean_value.values[index], df.clinical_visits_pre_24h_sbp_mean_value.values[index], df.exercise_value.values[index]])
            y_train.append(literal_eval(df.estimate_error[index]))
            MM_train.append(literal_eval(df.mm_post_finger_pressure_cycle[index]))
            wk_train.append(literal_eval(df.wk3_post_finger_pressure_cycle[index]))

    x_train = np.array(x_train)
    x_test = np.array(x_test)
    x_pred = np.array(x_pred)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    y_pred = np.array(y_pred)

    min_max_scaler = MinMaxScaler()
    x_train = min_max_scaler.fit_transform(x_train)
    x_test = min_max_scaler.transform(x_test)
    x_pred = min_max_scaler.transform(x_pred)

    x_train, y_train = zip(*random.sample(list(zip(list(x_train), y_train)), int(x_train.shape[0]*0.2)))
    x_train = np.array(x_train).reshape(len(x_train),len(x_train[0]))
    y_train = np.array(y_train).reshape(len(y_train),len(y_train[0]))
    
    return x_train, x_test, x_pred, y_train, y_test, y_pred, MM_train, MM_test, MM_pred, wk_train, wk_test, wk_pred, id_pred, id_test

def create_model():
    ### FUNCTIONAL API MODEL ###
    inputs = keras.Input(shape=(7,))
    dense1 = layers.Dense(50, activation="relu", use_bias=True)(inputs)
    dense2 = layers.Dense(100, activation="relu", use_bias=True)(dense1)
    dense3 = layers.Dense(150, activation="relu", use_bias=True)(dense2)
    outputs = layers.Dense(100)(dense3)
    model = keras.Model(inputs=inputs, outputs=outputs)

    model.compile(loss=loss_function, optimizer=optimizer)

    # model.summary()

    return model

def create_model_w_weights(tpers):
    inputs = keras.Input(shape=(7,))
    dense1 = layers.Dense(50, activation="relu", use_bias=True)(inputs)
    dense2 = layers.Dense(100, activation="relu", use_bias=True)(dense1)
    dense3 = layers.Dense(150, activation="relu", use_bias=True)(dense2)
    outputs = layers.Dense(100)(dense3)
    model = keras.Model(inputs=inputs, outputs=outputs)

    model.compile(loss=loss_function, optimizer=optimizer)
    model.load_weights("Data/residual_weights_pat_" + tpers)

    return model

def fit_model(model, x_train, y_train, x_val, y_val):
    history = model.fit(x_train, y_train, epochs=no_epochs, batch_size=batch_size,validation_data=(x_val, y_val), verbose=0)

    # Evaluate the model
    train_mse = model.evaluate(x_train, y_train, verbose=0)
    test_mse = model.evaluate(x_val, y_val, verbose=0)
    print('Train loss: %.3f, Test: %.3f' % (train_mse, test_mse))


    # Plot loss during training
    # plt.title('Loss / Mean Squared Error')
    # plt.plot(history.history['loss'], label='train')
    # plt.plot(history.history['val_loss'], label='test')
    # plt.legend()
    # plt.show()

    return model

def predict(x, y, wk, MM, model, plot):
    total_cycle_error = []
    dbp_error = []
    sbp_error = []
    pp_error = []
    MAP_error = []
    point_error = []
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for pers in range(len(x)):
        test_pers_x = x[pers] 
        test_pers_x = np.array(test_pers_x)
        test_pers = test_pers_x.reshape(1, -1)

        prediction = model.predict(test_pers)
        adding_prediction = [a + b for a, b in zip(prediction[0], wk[pers])]

        if plot == 1:
            plt.plot(prediction[0], 'g', linewidth=2)
            plt.plot(y[pers],'b')
            plt.show()

            plt.plot(adding_prediction, color='darkorange', label = 'Prediction')
            plt.plot(MM[pers], color='midnightblue', label= 'True curve')
            plt.plot(wk[pers], color='crimson', label= 'Windkessel model estimate')
            plt.legend()
            plt.title('Residual model with synthetic data')
            plt.xlabel('Time points [-]')
            plt.ylabel('Blood Pressure [mmHg]')
            plt.show()


        # Calculate erorrs
        dbp = min(adding_prediction)
        sbp = max(adding_prediction)
        pp = sbp-dbp
        MAP = np.mean(adding_prediction)
        dbp_true = min(MM[pers])
        sbp_true = max(MM[pers])
        pp_true = sbp_true-dbp_true
        MAP_true = np.mean(MM[pers])

        point_error.append(abs(prediction[0]-y[pers]))
        total_cycle_error.append(sum(abs(prediction[0]-y[pers])))
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
    print('Point error = ', np.mean(point_error), np.std(point_error))
    print('DBP error = ', np.mean(dbp_error), np.std(dbp_error))
    print('SBP error = ', np.mean(sbp_error), np.std(sbp_error))
    print('PP error = ', np.mean(pp_error), np.std(pp_error))
    print('MAP error = ', np.mean(MAP_error), np.std(MAP_error))
    print('Total error = ', np.mean(total_cycle_error), np.std(total_cycle_error))
    print('True positives = ', TP)
    print('False positives = ', FP)
    print('True negatives = ', TN)
    print('False negatives = ', FN)

    return np.mean(total_cycle_error)

x_train, x_test, x_pred, y_train, y_test, y_pred, MM_train, MM_test, MM_pred, wk_train, wk_test, wk_pred, id_pred, id_test = get_data()
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.10 , shuffle=True)


## RUN SEVERAL TIMES AND SAVE THE WEIGHTS FOR THE BEST PERFORMING MODEL ##
'''
best_error = 5000
for i in range(20):
    model_create = create_model()
    model_fitted = fit_model(model_create, x_train, y_train, x_val, y_val)
    error = predict(x_test, y_test, wk_test, MM_test, model_fitted, 0)
    print('Error for iteration ', i, ': ', error)
    if error < best_error:
        print('New best error on number ', i,'. Error = ', error)
        model_fitted.save_weights('Data/residual_weights_pat_' + str(id_test[0]))
        best_error = error
print('The best error achieved was: ', best_error)

'''

## EVALUATE ON PERSON LEFT OUT ## 
print('Model with weights from: ', id_test[0])
print('Predicting on person with trial ID: ', id_pred[0])
model = create_model_w_weights(str(id_test[0]))
#model = fit_model(model,x_train,y_train,x_val,y_val)
predict(x_pred, y_pred, wk_pred, MM_pred, model, 1)



