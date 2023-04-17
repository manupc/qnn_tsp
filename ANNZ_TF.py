#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 10:48:29 2023

@author: manupc
"""

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import pandas as pd
from scipy.stats import linregress
import time
import pickle
print('TF: ', tf.__version__)


def henon(length=10000, x0=None, a=1.4, b=0.3, discard=500):
    """Generate time series using the Henon map.
    Generates time series using the Henon map.
    Parameters
    ----------
    length : int, optional (default = 10000)
        Length of the time series to be generated.
    x0 : array, optional (default = random)
        Initial condition for the map.
    a : float, optional (default = 1.4)
        Constant a in the Henon map.
    b : float, optional (default = 0.3)
        Constant b in the Henon map.
    discard : int, optional (default = 500)
        Number of steps to discard in order to eliminate transients.
    Returns
    -------
    x : ndarray, shape (length, 2)
        Array containing points in phase space.
    """
    x = np.empty((length + discard, 2))

    if not x0:
        x[0] = (0.0, 0.9) + 0.01 * (-1 + 2 * np.random.random(2))
    else:
        x[0] = x0

    for i in range(1, length + discard):
        x[i] = (1 - a * x[i - 1][0] ** 2 + b * x[i - 1][1], x[i - 1][0])

    return x[discard:]

def mackey_glass(length=10000, x0=None, a=0.2, b=0.1, c=10.0, tau=23.0,
                 n=1000, sample=0.46, discard=250):
    """Generate time series using the Mackey-Glass equation.
    Generates time series using the discrete approximation of the
    Mackey-Glass delay differential equation described by Grassberger &
    Procaccia (1983).
    Parameters
    ----------
    length : int, optional (default = 10000)
        Length of the time series to be generated.
    x0 : array, optional (default = random)
        Initial condition for the discrete map.  Should be of length n.
    a : float, optional (default = 0.2)
        Constant a in the Mackey-Glass equation.
    b : float, optional (default = 0.1)
        Constant b in the Mackey-Glass equation.
    c : float, optional (default = 10.0)
        Constant c in the Mackey-Glass equation.
    tau : float, optional (default = 23.0)
        Time delay in the Mackey-Glass equation.
    n : int, optional (default = 1000)
        The number of discrete steps into which the interval between
        t and t + tau should be divided.  This results in a time
        step of tau/n and an n + 1 dimensional map.
    sample : float, optional (default = 0.46)
        Sampling step of the time series.  It is useful to pick
        something between tau/100 and tau/10, with tau/sample being
        a factor of n.  This will make sure that there are only whole
        number indices.
    discard : int, optional (default = 250)
        Number of n-steps to discard in order to eliminate transients.
        A total of n*discard steps will be discarded.
    Returns
    -------
    x : array
        Array containing the time series.
    """
    sample = int(n * sample / tau)
    grids = n * discard + sample * length
    x = np.empty(grids)

    if not x0:
        x[:n] = 0.5 + 0.05 * (-1 + 2 * np.random.random(n))
    else:
        x[:n] = x0

    A = (2 * n - b * tau) / (2 * n + b * tau)
    B = a * tau / (2 * n + b * tau)

    for i in range(n - 1, grids - 1):
        x[i + 1] = A * x[i] + B * (x[i - n] / (1 + x[i - n] ** c) +
                                   x[i - n + 1] / (1 + x[i - n + 1] ** c))
    return x[n * discard::sample]

def LoadData(T=7, zone=1, raw=False):

    if zone == 1:
        data= np.loadtxt('laser.dat').squeeze()
    elif zone==2:
        data = henon()[:, 0]
    elif zone == 3:
        data= mackey_glass(tau=7.0, sample=1.5, n=10)
    data= np.array(data)[-150:]
    if raw:
        return data
    X= []
    Y= []
    for i in range(len(data)-T):
        X.append(data[i:(i+T)])
        Y.append(data[i+T])
    X= np.array(X)
    Y= np.array(Y)
    

    X= 2*(X-X.min())/(X.max()-X.min())-1
    Y= 2*(Y-Y.min())/(Y.max()-Y.min())-1
    
    NTr= int(0.75*len(X))
    
    XTr= X[:NTr]
    YTr= Y[:NTr]
    XTs= X[NTr:]
    YTs= Y[NTr:]
    return XTr, YTr, XTs, YTs



###################################################################
# Multilayer FeedForward neural network model
###################################################################
class FFNetwork(tf.keras.Model):
    
    # Constructor.
    # Inputs:
    #    n_inputs: Number of network inputs
    #    n_outputs: Number of network outputs
    #    layers: List of pairs [#neurons, activation] for each non-linear layer, e.g. [[30, 'relu']]
    def __init__(self, n_inputs, n_outputs, layers):
        super(FFNetwork, self).__init__()
        self.n_inputs= n_inputs
        self.n_outputs= n_outputs
        self.capas= layers
        
        # Create the input layer
        input_layer = tf.keras.layers.InputLayer(input_shape=(n_inputs,))
        
        # Create intermediate layers
        sequence= []
        for layer in layers:
            
            activ= 'linear' if layer[1] is None else layer[1]
            
            currentLayer= tf.keras.layers.Dense(layer[0], activation=activ)
            sequence.append(currentLayer)
        
        # Create output layer
        output_layer= tf.keras.layers.Dense(n_outputs, activation='linear')
        
        
        
        # Pack everything as a sequence
        self.net= tf.keras.Sequential()
        self.net.add(input_layer)
        for layer in sequence:
            self.net.add(layer)
        self.net.add(output_layer)
        
        # Create weights
        self(tf.random.normal(shape=(1,self.n_inputs)))
        
        
    # forward pass for input batch x
    def call(self, x):
        x= tf.convert_to_tensor(x, dtype=tf.float32)
        return self.net(x)



for zone in range(1, 4):

    XTr, YTr, XTs, YTs= LoadData(T=7, zone=zone)
    
    
    # Time horizon
    T= XTr.shape[1]
    
    resultsFile= 'ANNresZ{}.pkl'.format(zone) # None for not saving results data
    lr= 0.01 # Learning rate of training algorithm
    layers= [[10, 'tanh']] # Description (number of neurons and activation function) of intermediate layers
    
    MaxEpochs= 30 # Maximum number of epochs to run the algorithm
    BatchSize= 64#32 # Batch size of training
    MaxExecutions= 30 # Number of experiments
    kFoldSplits= 4 # value of K for k-fold cross-validation
    
    
    
    
    valMSE= [] # History of MSE in the validation dataset for each fold
    testMSE= [] # History of MSE in the test data for each fold
    trainMSE= [] # History of MSE in the training data for each fold
    foldTrOuts= [] # Training outputs for each fold in the complete training dataset
    foldValOuts= [] # Validation outputs for each fold in the complete validation dataset
    valPred= []
    Times= []
    
    # Transfor validation dataset to tensor
    valIn= tf.convert_to_tensor(XTs, dtype='float32')
    valOut= tf.convert_to_tensor(YTs, dtype='float32')
    
    # Iterate for MaxExecutions runnings of the algorithm
    for execution in range(1, MaxExecutions+1):
    
        # FFN model definition
        model= FFNetwork(T, 1, layers)
    
        # Compile the model to optimize and track MSE with the RMSProp algorithm
        model.compile(loss=tf.keras.losses.MSE,
                  optimizer=tf.optimizers.RMSprop(learning_rate= lr),
                  metrics=['mean_squared_error'])
        t0= time.time()
        for epoch in range(MaxEpochs):
        
            # Iterate over folds
            currentFold= 0
            testDataPerSplit= int(np.ceil(len(XTr)/(kFoldSplits+1)))
            for i in range(kFoldSplits):
            
                currentFold+= 1 # Update current fold
    
                # Get training and test data
                trLastIdx= (i+1)*testDataPerSplit
                tsLastIdx= min((i+2)*testDataPerSplit, len(XTr)) 
    
                
                # Get training and test input/output patterns as tensors
                trIn= tf.convert_to_tensor(XTr[(i*testDataPerSplit):trLastIdx], dtype='float32')
                trOut= tf.convert_to_tensor(YTr[(i*testDataPerSplit):trLastIdx], dtype='float32')
    
                tsIn= tf.convert_to_tensor(XTr[trLastIdx:tsLastIdx], dtype='float32')
                tsOut= tf.convert_to_tensor(YTr[trLastIdx:tsLastIdx], dtype='float32')
    
                # fit the model
                print('Training exp. {}, epoch {}, fold {}...'.format(execution, epoch+1, currentFold), end='\r')
                model.fit(trIn, trOut, batch_size=BatchSize, epochs=1, verbose=0)
    
                # Generate generalization metrics
                scoresTr = model.evaluate(trIn, trOut, verbose=0)
                scoresTest = model.evaluate(tsIn, tsOut, verbose=0)
        tfin= time.time()
        scoresVal = model.evaluate(valIn, valOut, verbose=0)
    
        # Outputs
        foldOuts= model(XTr)
        foldTrOuts.append(foldOuts)
    
        foldVals= model(XTs)
        foldValOuts.append(foldVals)
    
        trainMSE.append(scoresTr[1])
        testMSE.append(scoresTest[1])
        valMSE.append(scoresVal[1])
        Times.append(tfin-t0)
        
        pred= model.predict(valIn)
        valPred.append(pred)
    
        print('Execution {}, MSE tr {:.3f}, ts {:.3f} val {:.3f}'.format(execution, trainMSE[-1], testMSE[-1], valMSE[-1]))
    
        # Save results to file
        if resultsFile is not None:
            storedData= (trainMSE, testMSE, valMSE, Times, valPred)
            with open(resultsFile, 'wb') as handle:
                pickle.dump(storedData, handle, protocol=pickle.HIGHEST_PROTOCOL)
