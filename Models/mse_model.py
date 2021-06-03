import os
from timeit import default_timer as timer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from ast import literal_eval
import my_functions
from sklearn.linear_model import LinearRegression
from itertools import chain
from tensorflow import keras
import tensorflow_probability as tfp
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import SGD, Adam
import tensorflow as tf
np.set_printoptions(threshold=np.inf)

def get_data():
    df = pd.read_csv('Data/full_data_set_double.csv', header=0)
    full = pd.read_csv('Data/full_data_set.csv', header=0)
    pre_ts = []
    post_ts = []
    all_ts = []
    q_ts = []
    input_var = []
    T_s = []

    for index, row in df.iterrows():
        input_variables = []
        if pd.isna(row['clinical_visits_post_24h_dbp_mean_value']) or pd.isna(row['clinical_visits_post_24h_sbp_mean_value']):
            continue
        if pd.isna(row['clinical_visits_pre_24h_sbp_mean_value']) or pd.isna(row['clinical_visits_pre_24h_dbp_mean_value']):
            continue
        if sum(literal_eval(df.post_finger_pressure_cycle[index])) == 0:
            continue
        if sum(literal_eval(df.post_flow[index])) == 0:
            continue
        # Drop curves that are of vary low quality
        if index == 8 or index == 38 or index == 39:
            continue
        else:
            pre_ts.append(literal_eval(df.pre_finger_pressure_cycle[index]))
            post_ts.append(literal_eval(df.post_finger_pressure_cycle[index]))
            all_ts.append(literal_eval(df.pre_finger_pressure_cycle[index]))
            all_ts.append(literal_eval(df.post_finger_pressure_cycle[index]))
            q_ts.append(literal_eval(df.post_flow[index]))
            T_s.append(df.true_time_post_flow[index])
            
            age = float(df.roottable_age_value[index])
            gender = float(df.roottable_sex_item[index])
            PAI = float(df.MeanPAIPerDay[index])
            bmi = float(df.clinical_visits_body_mass_index_value[index])
            vo2 = float(df.clinical_visits_cpet_vo2max_value[index])
            pre_dbp = float(df.clinical_visits_pre_24h_dbp_mean_value[index])
            pre_sbp = float(df.clinical_visits_pre_24h_sbp_mean_value[index])
            period = float(df.true_time_post_flow[index])
            input_variables.append(age)
            input_variables.append(gender)
            input_variables.append(bmi)
            input_variables.append(vo2)
            input_variables.append(pre_dbp)
            input_variables.append(pre_sbp)
            input_variables.append(PAI)

            input_var.append(input_variables) 

    pre_ts = np.array(pre_ts) # all pre pressure curves
    post_ts = np.array(post_ts) # all post pressure curves
    all_ts = np.array(all_ts) # all pressure curves
    q_ts = np.array(q_ts) # all post flow curves

    return input_var, post_ts, q_ts, T_s

def process_data(x_all, y_all, q_all, t_all):

    times = []
    for time in t_all:
        ts = []
        for x in range(100):
            ts.append(time)
        times.append(ts)
    times = np.array(times)

    # Normalize
    scaler = preprocessing.MinMaxScaler()
    full_x = scaler.fit_transform(x_all)

    y = []
    for i in range(0,y_all.shape[0],100):
        y.append(y_all[i:i+100])
    y = y[0]

    x = []
    for i in range(0,full_x.shape[0],100):
        x.append(full_x[i:i+100])
    x = x[0]

    q = []
    for i in range(0,q_all.shape[0],100):
        q.append(q_all[i:i+100])
    q = q[0]

    t = []
    for i in range(0,times.shape[0],100):
        t.append(times[i:i+100])
    t = t[0]

    dataset = tf.data.Dataset.from_tensor_slices((x, y, q, t))
    eval_dataset = dataset.take(3) 
    train_dataset = dataset.skip(3)

    batch_size = 1
    train_dataset = train_dataset.shuffle(buffer_size=40).batch(batch_size)
    eval_dataset = eval_dataset.shuffle(buffer_size=40).batch(batch_size)

    return train_dataset, eval_dataset

def create_model():
    inputs = keras.Input(shape=(None,7), name="features_components")
    x1 = layers.Dense(25, activation="relu", kernel_initializer="glorot_uniform",kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4))(inputs)
    x2 = layers.Dense(50, activation="relu",kernel_initializer="glorot_uniform",kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4))(x1)
    outputs = layers.Dense(100, name="predictions")(x2)
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model

loss_history = []

def train_step(model,x,labels,q,t):
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    loss_object = keras.losses.MeanSquaredError()
    a = 30
    b = 1
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        logits = tf.reshape(logits,(1,100,1))

        # Add asserts to check the shape of the output.
        tf.debugging.assert_equal(logits.shape, (1, 100,1))
        mse_loss = loss_object(labels, logits)
        mse_loss = tf.dtypes.cast(mse_loss, tf.float64)

        loss_value = mse_loss

    loss_history.append(loss_value.numpy().mean())
    grads = tape.gradient(loss_value, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    
def train(model, epochs):
    start_time = timer()
    for epoch in range(epochs):
        for (batch, (x,labels,q,t)) in enumerate(train_dataset):
            train_step(model,x,labels, q,t)
        print('Epoch {} finished'.format(epoch))
    end_time = timer()
    print('Training time: ',int((end_time-start_time)/60), ' minutes and ', (end_time-start_time)%60, ' seconds' )

def evaluate(pred, true):
    dbp = min(pred)
    sbp = max(pred)
    pp = sbp-dbp
    MAP = np.mean(pred)
    dbp_true = min(true)
    sbp_true = max(true)
    pp_true = sbp_true-dbp_true
    MAP_true = np.mean(true)

    total_cycle_error = sum(abs(pred-true))
    dbp_error = abs(dbp-dbp_true)
    sbp_error = abs(sbp-sbp_true)
    pp_error = abs(pp-pp_true)
    MAP_error = abs(MAP-MAP_true)
    TP = 0
    TN = 0
    FP = 0
    FN = 0
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

    plt.plot(pred, color='darkorange', label = 'Prediction')
    plt.plot(true, color='midnightblue', label= 'True curve')
    plt.legend()
    plt.title('Model with MSE loss')
    plt.xlabel('Time points [-]')
    plt.ylabel('Blood Pressure [mmHg]')
    plt.show()
    print('DBP error = ', np.mean(dbp_error))
    print('SBP error = ', np.mean(sbp_error))
    print('PP error = ', np.mean(pp_error))
    print('MAP error = ', np.mean(MAP_error))
    print('Total error = ', np.mean(total_cycle_error))
    print('True positives = ', TP)
    print('False positives = ', FP)
    print('True negatives = ', TN)
    print('False negatives = ', FN)
    print('\n')



x, y, q, t = get_data()
train_dataset, eval_dataset = process_data(x, y, q, t)
model = create_model()
train(model,epochs = 100)

for (x,y,q,t) in list(eval_dataset.as_numpy_iterator()):
    pred = model(x)
    evaluate(pred.numpy().flatten(),y[0])


plt.plot(loss_history)
plt.xlabel('Batch #')
plt.ylabel('Loss [MSE]')
plt.show()
quit()