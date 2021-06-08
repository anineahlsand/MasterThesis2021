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
np.set_printoptions(threshold=np.inf)

df = pd.read_csv('Data/residual_dataset_new.csv', header=0)
df.reset_index(drop=True,inplace=True)

# Model configuration
batch_size = 5
loss_function = 'mean_squared_error'
no_epochs = 100
optimizer = Adam(lr=0.001)
ids = []
testing_pers =[]
testing_lables = []
testing_post_MM = []
testing_post_true = [] 
testing_id = []
evaluating_pers = []
evaluating_lables = []
evaluating_post_MM = []
evaluating_post_true = []
evaluating_id = []

def get_data():

    training_set = []
    lables = []
    post_ts_MM = []
    post_ts_true = []
    pre_ts_true = []

    for index, row in df.iterrows():
        if sum(literal_eval(df.true_post_finger_pressure_cycle[index])) == 0:
            continue
        if (df.roottable_case_id_text.values[index]) == 217:
            testing_pers.append([df.roottable_age_value.values[index], df.roottable_sex_item.values[index], df.clinical_visits_body_mass_index_value.values[index], df.clinical_visits_cpet_vo2max_value.values[index],
                            df.clinical_visits_pre_24h_dbp_mean_value.values[index], df.clinical_visits_pre_24h_sbp_mean_value.values[index], df.exercise_value.values[index]])
            testing_lables.append(literal_eval(df.estimate_error[index]))
            testing_post_MM.append(literal_eval(df.mm_post_finger_pressure_cycle[index]))
            testing_post_true.append(literal_eval(df.true_post_finger_pressure_cycle[index]))
            testing_id.append(df.roottable_case_id_text.values[index])
        elif (df.roottable_case_id_text.values[index]) == 67:
            evaluating_pers.append([df.roottable_age_value.values[index], df.roottable_sex_item.values[index], df.clinical_visits_body_mass_index_value.values[index], df.clinical_visits_cpet_vo2max_value.values[index],
                            df.clinical_visits_pre_24h_dbp_mean_value.values[index], df.clinical_visits_pre_24h_sbp_mean_value.values[index], df.exercise_value.values[index]])
            evaluating_lables.append(literal_eval(df.estimate_error[index]))
            evaluating_post_MM.append(literal_eval(df.mm_post_finger_pressure_cycle[index]))
            evaluating_post_true.append(literal_eval(df.true_post_finger_pressure_cycle[index]))
            evaluating_id.append(df.roottable_case_id_text.values[index])
        else:
            training_set.append([df.roottable_age_value.values[index], df.roottable_sex_item.values[index], df.clinical_visits_body_mass_index_value.values[index], df.clinical_visits_cpet_vo2max_value.values[index],
                            df.clinical_visits_pre_24h_dbp_mean_value.values[index], df.clinical_visits_pre_24h_sbp_mean_value.values[index], df.exercise_value.values[index]])
            lables.append(literal_eval(df.estimate_error[index]))
            post_ts_MM.append(literal_eval(df.mm_post_finger_pressure_cycle[index]))
            post_ts_true.append(literal_eval(df.true_post_finger_pressure_cycle[index]))
            pre_ts_true.append(literal_eval(df.pre_finger_pressure_cycle[index]))
            ids.append(df.roottable_case_id_text.values[index])

    x_values = pd.DataFrame(training_set).values
    y_all = lables 

    ts = [] 
    ts.extend(pre_ts_true)
    ts.extend(post_ts_true)
    ts = np.array(ts)
    mean_ts = my_functions.make_mean_vector(ts)
    ts_wo_mean = my_functions.subtract_mean_from_post_ts_data(ts, mean_ts)
    mean_vector_pre = my_functions.make_mean_vector(pre_ts_true)
    pre_ts_wo_mean = my_functions.subtract_mean_from_post_ts_data(pre_ts_true, mean_vector_pre)

    # PCA
    pca_model = PCA(0.95)
    pca_model.fit(ts_wo_mean)
    loadings = pca_model.transform(pre_ts_wo_mean) # number of components: 4

    min_max_scaler = MinMaxScaler()
    x_norm = min_max_scaler.fit_transform(x_values)
    y_all = np.array(y_all)

    train_x = np.array(x_norm)
    train_y = np.array(y_all)

    return train_x, train_y, post_ts_MM, post_ts_true, min_max_scaler 
train_x, train_y, post_ts_MM, post_ts_true, min_max_scaler = get_data()
print('Predicting for person with trial ID: ', evaluating_id[0])

def create_model():
    ### FUNCTIONAL API MODEL ###
    inputs = keras.Input(shape=(7,))
    dense1 = layers.Dense(50, activation="relu")(inputs)
    dense2 = layers.Dense(100, activation="relu")(dense1)
    dense3 = layers.Dense(150, activation="relu")(dense2)
    outputs = layers.Dense(100)(dense3)
    model = keras.Model(inputs=inputs, outputs=outputs)

    model.compile(loss=loss_function, optimizer=optimizer)

    return model
# model = create_model()

def fit_model(model):
    history = model.fit(train_x, train_y, epochs=no_epochs, batch_size=batch_size, verbose=0)

    # Evaluate the model
    # train_mse = model.evaluate(train_x, train_y, verbose=0)
    # test_mse = model.evaluate(test_x, test_y, verbose=0)
    # print('Train loss: %.3f, Test: %.3f' % (train_mse, test_mse))

    return model
# model = fit_model()

def create_model_saved_weights():
    inputs = keras.Input(shape=(7,))
    dense1 = layers.Dense(50, activation="relu")(inputs)
    dense2 = layers.Dense(100, activation="relu")(dense1)
    dense3 = layers.Dense(150, activation="relu")(dense2)
    outputs = layers.Dense(100)(dense3)
    model = keras.Model(inputs=inputs, outputs=outputs)

    model.compile(loss=loss_function, optimizer=optimizer)

    model.load_weights("Data/residual_weights_real_pat_" + str(evaluating_id[0]))

    # model.summary()

    return model

## TESTING ##
def predict(predict_pers_x, predict_pers_y, model, post_ts_MM, post_ts_true, i):
    predict_pers_xx = predict_pers_x.reshape(predict_pers_x.shape[0], 1)
    prediction = model.predict(predict_pers_xx.T)
    
    adding_prediction = [b - a for a, b in zip(prediction[0], post_ts_MM[0])]

    if i == 1:
        plt.plot(prediction[0], 'g')
        plt.plot(predict_pers_y[0], 'b')
        plt.show()

        plt.plot(adding_prediction, color='darkorange', label = 'Prediction')
        plt.plot(post_ts_true[0], color='midnightblue', label= 'True curve')
        plt.plot(post_ts_MM[0], color='crimson', label='Mechanistic model estimate')
        plt.legend()
        plt.title('Residual model with real data')
        plt.xlabel('Time points [-]')
        plt.ylabel('Blood pressure [mmHg]')
        plt.gcf().set_dpi(200)
        plt.show()

    # Calculate erorrs
    dbp = min(adding_prediction)
    sbp = max(adding_prediction)
    pp = sbp-dbp
    MAP = np.mean(adding_prediction)
    dbp_true = min(post_ts_true[0])
    sbp_true = max(post_ts_true[0])
    pp_true = sbp_true-dbp_true
    MAP_true = np.mean(post_ts_true[0])

    point_error = abs(prediction[0]-predict_pers_y[0])
    total_cycle_error = sum(abs(prediction[0]-predict_pers_y[0]))
    dbp_error = abs(dbp-dbp_true)
    sbp_error = abs(sbp-sbp_true)
    pp_error = abs(pp-pp_true)
    MAP_error = abs(MAP-MAP_true)

    print('Point error = ', np.mean(point_error), np.std(point_error))
    print('DBP error = ', np.mean(dbp_error), np.std(dbp_error))
    print('SBP error = ', np.mean(sbp_error), np.std(sbp_error))
    print('PP error = ', np.mean(pp_error), np.std(pp_error))
    print('MAP error = ', np.mean(MAP_error), np.std(MAP_error))
    print('Total error = ', np.mean(total_cycle_error), np.std(total_cycle_error))

    return np.mean(total_cycle_error)
# error = predict()


## RUN SEVERAL TIMES AND SAVE THE BEST MODEL ##
'''
best_error = 4000
evaluating_pers = np.array(evaluating_pers) 
evaluating_pers = min_max_scaler.transform(evaluating_pers)
for i in range(30):
    model_eval = create_model()
    model_fitted = fit_model(model_eval)
    error = predict(evaluating_pers.T, evaluating_lables, model_fitted, evaluating_post_MM, evaluating_post_true, 0)
    print('Iteration: ', i, ' with error: ', error)
    if error < best_error:
        print('New best error on number ', i,'. Error = ', error)
        model_fitted.save_weights('Data/residual_weights_real_pat_' + str(evaluating_id[0]))
        best_error = error
print('The best achieved error for ', evaluating_id[0], ' was: ', best_error)
'''

## EVALUATE ON THE PERSON LEFT OUT ##
testing_pers = np.array(testing_pers) 
testing_pers = min_max_scaler.transform(testing_pers)
testing_lables = np.array(testing_lables)
model_test = create_model_saved_weights()
print('Testing on person with trial ID: ', testing_id[0])
print(testing_pers.T)
error1 = predict(testing_pers.T, testing_lables, model_test, testing_post_MM, testing_post_true, 1)
