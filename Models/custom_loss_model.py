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
# import tensorflow_probability as tfp
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import SGD, Adam
import tensorflow as tf
np.set_printoptions(threshold=np.inf)

def get_data():
    df = pd.read_csv('Data/full_data_set_double.csv', header=0)
    train_x = []
    train_y = []
    train_q = []
    train_t = []
    test_x = []
    test_y = []
    test_q = []
    test_t = []
    pred_x = []
    pred_y = []
    pred_q = []
    pred_t = []

    for index, row in df.iterrows():
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
        if df.roottable_case_id_text[index] == 367:
            if len(test_x) == 0:
                test_y.append(literal_eval(df.post_finger_pressure_cycle[index]))
                test_q.append(literal_eval(df.post_flow[index]))
                test_t.append(df.true_time_post_flow[index])
                test_x.append([float(df.roottable_age_value[index]),float(df.roottable_sex_item[index]),float(df.clinical_visits_body_mass_index_value[index]),float(df.clinical_visits_cpet_vo2max_value[index]),float(df.clinical_visits_pre_24h_dbp_mean_value[index]),float(df.clinical_visits_pre_24h_sbp_mean_value[index]),float(df.MeanPAIPerDay[index])])
            else:
                train_y.append(literal_eval(df.post_finger_pressure_cycle[index]))
                train_q.append(literal_eval(df.post_flow[index]))
                train_t.append(df.true_time_post_flow[index])
                train_x.append([float(df.roottable_age_value[index]),float(df.roottable_sex_item[index]),float(df.clinical_visits_body_mass_index_value[index]),float(df.clinical_visits_cpet_vo2max_value[index]),float(df.clinical_visits_pre_24h_dbp_mean_value[index]),float(df.clinical_visits_pre_24h_sbp_mean_value[index]),float(df.MeanPAIPerDay[index])])
                
        elif df.roottable_case_id_text[index] == 503:
            if len(pred_x) == 0:
                pred_y.append(literal_eval(df.post_finger_pressure_cycle[index]))
                pred_q.append(literal_eval(df.post_flow[index]))
                pred_t.append(df.true_time_post_flow[index])
                pred_x.append([float(df.roottable_age_value[index]),float(df.roottable_sex_item[index]),float(df.clinical_visits_body_mass_index_value[index]),float(df.clinical_visits_cpet_vo2max_value[index]),float(df.clinical_visits_pre_24h_dbp_mean_value[index]),float(df.clinical_visits_pre_24h_sbp_mean_value[index]),float(df.MeanPAIPerDay[index])])
            else:
                train_y.append(literal_eval(df.post_finger_pressure_cycle[index]))
                train_q.append(literal_eval(df.post_flow[index]))
                train_t.append(df.true_time_post_flow[index])
                train_x.append([float(df.roottable_age_value[index]),float(df.roottable_sex_item[index]),float(df.clinical_visits_body_mass_index_value[index]),float(df.clinical_visits_cpet_vo2max_value[index]),float(df.clinical_visits_pre_24h_dbp_mean_value[index]),float(df.clinical_visits_pre_24h_sbp_mean_value[index]),float(df.MeanPAIPerDay[index])])
        else:
            train_y.append(literal_eval(df.post_finger_pressure_cycle[index]))
            train_q.append(literal_eval(df.post_flow[index]))
            train_t.append(df.true_time_post_flow[index])
            train_x.append([float(df.roottable_age_value[index]),float(df.roottable_sex_item[index]),float(df.clinical_visits_body_mass_index_value[index]),float(df.clinical_visits_cpet_vo2max_value[index]),float(df.clinical_visits_pre_24h_dbp_mean_value[index]),float(df.clinical_visits_pre_24h_sbp_mean_value[index]),float(df.MeanPAIPerDay[index])])
    
    train_y = np.array(train_y)
    train_q = np.array(train_q)
    test_y = np.array(test_y)
    test_q = np.array(test_q)
    pred_y = np.array(pred_y)
    pred_q = np.array(pred_q)

    return train_x, train_y, train_q, train_t,test_x, test_y, test_q, test_t,pred_x, pred_y, pred_q, pred_t

def process_data(train_x, train_y, train_q, train_t,test_x, test_y, test_q, test_t,pred_x, pred_y, pred_q, pred_t):
    # Train T
    times_train = []
    for time in train_t:
        ts = []
        for x in range(100):
            ts.append(time)
        times_train.append(ts)
    times_train = np.array(times_train)
    # Test T
    times_test = []
    for time in test_t:
        ts = []
        for x in range(100):
            ts.append(time)
        times_test.append(ts)
    times_test = np.array(times_test)
    # Pred T
    times_pred = []
    for time in pred_t:
        ts = []
        for x in range(100):
            ts.append(time)
        times_pred.append(ts)
    times_pred = np.array(times_pred)

    # Normalize
    scaler = preprocessing.MinMaxScaler()
    x = scaler.fit_transform(train_x)
    x_t = scaler.transform(test_x)
    x_p = scaler.transform(pred_x)

    # Train data
    y_train = []
    for i in range(0,train_y.shape[0],100):
        y_train.append(train_y[i:i+100])
    y_train = y_train[0]

    x_train = []
    for i in range(0,x.shape[0],100):
        x_train.append(x[i:i+100])
    x_train = x_train[0]

    q_train = []
    for i in range(0,train_q.shape[0],100):
        q_train.append(train_q[i:i+100])
    q_train = q_train[0]

    t_train = []
    for i in range(0,times_train.shape[0],100):
        t_train.append(times_train[i:i+100])
    t_train = t_train[0]
    # Test data
    y_test = []
    for i in range(0,test_y.shape[0],100):
        y_test.append(test_y[i:i+100])
    y_test = y_test[0]

    x_test = []
    for i in range(0,x_t.shape[0],100):
        x_test.append(x_t[i:i+100])
    x_test = x_test[0]

    q_test = []
    for i in range(0,test_q.shape[0],100):
        q_test.append(test_q[i:i+100])
    q_test = q_test[0]

    t_test = []
    for i in range(0,times_test.shape[0],100):
        t_test.append(times_test[i:i+100])
    t_test = t_test[0]
    # Pred data   
    y_pred = []
    for i in range(0,pred_y.shape[0],100):
        y_pred.append(pred_y[i:i+100])
    y_pred = y_pred[0]

    x_pred = []
    for i in range(0,x_p.shape[0],100):
        x_pred.append(x_p[i:i+100])
    x_pred = x_pred[0]

    q_pred = []
    for i in range(0,pred_q.shape[0],100):
        q_pred.append(pred_q[i:i+100])
    q_pred = q_pred[0]

    t_pred = []
    for i in range(0,times_pred.shape[0],100):
        t_pred.append(times_pred[i:i+100])
    t_pred = t_pred[0]

    dataset_train = tf.data.Dataset.from_tensor_slices((x_train, y_train, q_train, t_train))
    dataset_test = tf.data.Dataset.from_tensor_slices((x_test, y_test, q_test, t_test))
    dataset_pred = tf.data.Dataset.from_tensor_slices((x_pred, y_pred, q_pred, t_pred))

    batch_size = 1
    dataset_train = dataset_train.shuffle(buffer_size=40).batch(batch_size)
    dataset_test = dataset_test.shuffle(buffer_size=40).batch(batch_size)
    dataset_pred = dataset_pred.shuffle(buffer_size=40).batch(batch_size)

    return dataset_train, dataset_test, dataset_pred

def create_model():
    inputs = keras.Input(shape=(None,7), name="features_components")
    x1 = layers.Dense(25, activation="relu", kernel_initializer="glorot_uniform",kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4))(inputs)
    x2 = layers.Dense(50, activation="relu",kernel_initializer="glorot_uniform",kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4))(x1)
    x3 = layers.Dense(75, activation="relu",kernel_initializer="glorot_uniform",kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4))(x2)
    outputs = layers.Dense(100, name="predictions")(x3)
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model

def my_loss(y_pred, q, t):
    time = t.numpy()[0][0]
    y_pred = tf.dtypes.cast(y_pred, tf.float64)
    N = 100 
    w = np.arange(N//2+1)*2*np.pi/time
    PP = tf.subtract(tf.reduce_max(y_pred,1),tf.reduce_min(y_pred,1))
    
    # Calculating integral of Q to find SO
    factor = q[0][0] + q[0][99]
    for i in range(1,100-1):
        factor += 2*q[0][i]
    SV = (time/100)*factor
    
    MAP = tf.reduce_mean(y_pred, axis=1)
    CO = SV / time
    R = MAP / CO
    C = PP / SV

    # Real and imaginary part of Z_model
    imag = -(tf.math.multiply(tf.math.multiply(tf.math.square(R),w),C))/(tf.math.multiply(tf.math.multiply(tf.math.square(R),tf.math.square(w)),tf.math.square(C))+1)
    real = R/(tf.math.multiply(tf.math.multiply(tf.math.square(R),tf.math.square(w)),tf.math.square(C))+1)

    z_model = tf.complex(real,imag)
    z_pred = tf.signal.rfft(tf.squeeze(y_pred,axis=2)) / tf.signal.rfft(q)
    z_diff = z_model-z_pred

    # Calculating magnitude of Z loss
    a = tf.math.real(z_diff)
    b = tf.math.imag(z_diff)
    z_mag = tf.math.sqrt(tf.math.square(a) + tf.math.square(b))
    z_sum = tf.math.reduce_sum(z_mag)

    total_loss = z_sum
   
    return total_loss

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
        custom_loss = my_loss(logits,q,t)
        mse_loss = loss_object(labels, logits)
        mse_loss = tf.dtypes.cast(mse_loss, tf.float64)

        loss_value = a * custom_loss + b * mse_loss

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

    point_error = abs(pred-true)
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
    plt.title('Model with custom loss')
    plt.xlabel('Time points [-]')
    plt.ylabel('Blood pressure [mmHg]')
    plt.gcf().set_dpi(200)
    plt.show()
    
    print('DBP error = ', np.mean(dbp_error))
    print('SBP error = ', np.mean(sbp_error))
    print('PP error = ', np.mean(pp_error))
    print('MAP error = ', np.mean(MAP_error))
    print('Point error = ', np.mean(point_error))
    print('True positives = ', TP)
    print('False positives = ', FP)
    print('True negatives = ', TN)
    print('False negatives = ', FN)
    print('\n')
    return np.mean(point_error)


train_x, train_y, train_q, train_t,test_x, test_y, test_q, test_t,pred_x, pred_y, pred_q, pred_t = get_data()
train_dataset, test_dataset, pred_dataset = process_data(train_x, train_y, train_q, train_t,test_x, test_y, test_q, test_t,pred_x, pred_y, pred_q, pred_t)
#model = create_model()
#train(model,epochs = 100)
'''
best_error = np.inf
for i in range(10):
    model = create_model()
    train(model,epochs = 100)
    for (x,y,q,t) in list(test_dataset.as_numpy_iterator()):
        pred = model(x)
        error = evaluate(pred.numpy().flatten(),y[0])
    if error < best_error:
        print('\n New best error on number ',i,'. Error = ', error, '\n')
        model.save_weights('Data/custom_weights_647_890')
        best_error = error
print('The best error is: ', best_error)
'''

for (x,y,q,t) in list(pred_dataset.as_numpy_iterator()):
    model = create_model()
    model.load_weights('Data/custom_weights_367_503')
    pred = model(x)
    error = evaluate(pred.numpy().flatten(),y[0])
    print('Final error: ', error)


plt.plot(loss_history)
plt.xlabel('Batch #')
plt.ylabel('Loss [MSE]')
plt.show()
quit()