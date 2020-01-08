"""
This code is mainly from https://www.kaggle.com/gianfrancobarone/lanl-nn-starter
"""
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import seismic_prediction as context

# Any results you write to the current directory are saved as output.

from sklearn.model_selection import train_test_split

#Libraries for neural net
import keras
from keras.layers import Dense
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import Dropout
from keras.models import Sequential
from keras import optimizers
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from sklearn.metrics import mean_absolute_error
from hyperas import distributions
import gc

import seismic_prediction as context

# global paras
Ytrain = 0
Xtrain = 0

def NN_model(preparation_threads=[]):
    """test the model"""
    
    for preparation_thread in preparation_threads:
        if preparation_thread != None:
            preparation_thread.join()

    # load samples
    global Xtrain, Ytrain
    train_features = pd.read_csv(context.saving_path+'train_features2.csv')
    test_features = pd.read_csv(context.saving_path+'test_features2.csv')
    train_features_denoised = pd.read_csv(context.saving_path+'train_features_denoised.csv')
    test_features_denoised = pd.read_csv(context.saving_path+'test_features_denoised.csv')
    train_features_denoised.columns = [(str(i)+'_denoised') for i in train_features_denoised.columns]
    test_features_denoised.columns = [(str(i)+'_denoised') for i in test_features_denoised.columns]
    Ytrain = pd.read_csv(context.saving_path+'y.csv')
    Xtrain = pd.concat([train_features, train_features_denoised], axis=1).drop(['seg_id_denoised', 'target_denoised'], axis=1)
    Xtest = pd.concat([test_features, test_features_denoised], axis=1).drop(['seg_id_denoised', 'target_denoised'], axis=1)

    submission = pd.read_csv(context.sample_submission_path+'sample_submission.csv', index_col='seg_id', dtype={"time_to_failure": np.float32})

    X_train, X_val, Y_train, Y_val = train_test_split(Xtrain, Ytrain, test_size=0.2, random_state=5)
    
    #Model parameters
    kernel_init = 'he_normal'
    input_size = len(Xtrain.columns)

    ### Neural Network ###

    # Model architecture: A very simple shallow Neural Network 
    model = Sequential()
    model.add(Dense(512, input_dim = input_size))
     
    model.add(Activation('relu'))
    model.add(BatchNormalization())
   #model.add(Dropout(0.7))
#    model.add(Dense(512))    
#    model.add(Activation('tanh'))
    #model.add(BatchNormalization())
    #model.add(Dropout(0.7))
    model.add(Dense(256))    
    model.add(Activation('tanh'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(1))    
    model.add(Activation('linear'))

    #compile the model
    optim = optimizers.Adam(lr = 0.005)
    model.compile(loss = 'mean_absolute_error', optimizer = optim)

    #Callbacks
    csv_logger = CSVLogger('log.csv', append=True, separator=';')
    best_model = ModelCheckpoint("/NN_model.hdf5", save_best_only=True, period=3)
    restore_best = EarlyStopping(monitor='val_loss', verbose=2, patience=100)# restore_best_weights=True)
    model.fit(x=X_train, y=Y_train, batch_size=64, epochs=2000, verbose=2, callbacks=[csv_logger, best_model], validation_data=(X_val,Y_val))

    ### Neural Network End ###
    model.load_weights('/NN_model.hdf5')

    # output CV score 
    nn_predictions = model.predict(X_train, verbose = 2, batch_size = 64)
    score = mean_absolute_error(Y_train, nn_predictions)
    with open(context.saving_path+'logs.txt', 'a') as f: 
        print("CV score of NN is %f"%(score,), file=f)

    # output test predictions
    nn_predictions = model.predict(Xtest, verbose = 2, batch_size = 64)
    submission['time_to_failure'] = nn_predictions
    submission.to_csv(context.saving_path+'NN_submission.csv')

from sklearn.preprocessing import Imputer
def NN_model_hr(X_train, X_val, Y_train, Y_val, paras, hidden_layer_num=3):
    """be function for hyperopt, actually hyperas take a lot memory"""
    
    #Model parameters
    kernel_init = 'he_normal'
    input_size = X_train.shape[1]

    ### Neural Network ###

    # Model architecture: A very simple shallow Neural Network 
    model = Sequential()
    
    model.add(Dense(paras["Dense1_size"]+100, input_dim = input_size)) #16 
    model.add(Activation(paras["Activation1"])) #'linear'
    if paras["BatchNorm1"] == True:
        model.add(BatchNormalization())
    model.add(Dropout(paras["Dropout_rate1"])) #0.5

    # hidden layers
    for i in range(hidden_layer_num):
        model.add(Dense(paras["Dense"+str(i+2)+"_size"])) #32
        model.add(Activation(paras["Activation"+str(i+2)])) #'tanh'
        if paras["BatchNorm"+str(i+2)] == True:
            model.add(BatchNormalization())
        model.add(Dropout(paras["Dropout_rate"+str(i+2)])) #0.5

    model.add(Dense(1))  
    model.add(Activation('linear'))

    #compile the model
    optim = paras['optimizer'](lr = paras['learn_rate'])  #optimizers.Adam(lr = 0.001)
    model.compile(loss = 'mean_absolute_error', optimizer = optim)

    #Callbacks
    csv_logger = CSVLogger('log.csv', append=True, separator=';')
    best_model = ModelCheckpoint("/NN_model.hdf5", save_best_only=True, period=1)
    restore_best = EarlyStopping(monitor='val_loss', verbose=False, patience=100, restore_best_weights=True)
    model.fit(x=X_train, y=Y_train, batch_size=64, epochs=40, verbose=False, callbacks=[csv_logger, best_model], validation_data=(X_val,Y_val))

    ### Neural Network End ###
    try:
        model.load_weights('/NN_model.hdf5')
        os.remove('/NN_model.hdf5')
    except:
        # performance may not decrease
        nothing = None

    # output CV score 
    nn_predictions = model.predict(X_val, verbose = False, batch_size = 64)
    
    # calculate a score
    try:
        score = mean_absolute_error(Y_val, nn_predictions)
    except:
        nn_predictions = np.array(Y_val + 99)   # it means failure
        score = mean_absolute_error(Y_val, nn_predictions)
    print("CV score of NN is %f"%(score,))

    gc.collect()
    
    # return predictions on X_val
    return nn_predictions