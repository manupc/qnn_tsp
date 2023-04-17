#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 10:38:18 2023

@author: manupc
"""

import numpy as np
import tensorflow as tf
import tensorflow_quantum as tfq
import cirq
import sympy
from matplotlib import pyplot as plt
import pandas as pd
import time
import pickle
print('TF: ', tf.__version__)
print('TFQ: ', tfq.__version__)



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
# Tensorflow Quantum layer to create the quantum circuit of the VQC
###################################################################
class QuantumLayer(tf.keras.layers.Layer):
    
    # Constructor.
    # Inputs:
    #   qubits: The set of qubits used in the quantum circuit (of length T)
    #   L: Number of layers
    #   autoScaleInputs: True to weight input data to the circuit; false to avoid this step
    def __init__(self,qubits, L, autoScaleInputs= False, name="QuantumLayer"):
        super(QuantumLayer, self).__init__(name=name)
        
        # Set object variables
        self.qubits= qubits
        self.T= len(qubits)
        self.L= L
        self.autoScaleInputs= autoScaleInputs
        
        
        # Create the circuit and get the circuit object and the name of input and parameter circuit data
        self.circuit, self.inSym, self.WSym= self.createCircuit(qubits, L, self.T) 

        # Create the observable to be measured
        aux= cirq.Z(self.qubits[0])
        #for i in range(1, self.T):
        #    aux= aux + cirq.Z(self.qubits[i])
        
        self.observables= [aux]
        
        # Quantum circuit parameters (for rotation gates)
        W_init = tf.random_uniform_initializer(minval=0.0, maxval=2*np.pi)  # Random weight initialization in [0, 2pi]
        self.W = tf.Variable( initial_value=W_init(shape=(1, len(self.WSym)), dtype="float32"), trainable=True, name="W")
        
        # If autoScaleInputs is True, then create weights for classic input data
        if autoScaleInputs:
            Wi_init = tf.ones(shape=(self.T,))
            self.Wi = tf.Variable(initial_value=Wi_init, dtype="float32", trainable=True, name="Winputs")
        else:
            self.Wi= None

        # Sort symbols to set W and Wi properly
        symbols = [str(symb) for symb in self.WSym + self.inSym]
        self.indices = tf.constant([symbols.index(a) for a in sorted(symbols)])
        
        # Create the tensorflow computation layer for the circuit
        self.empty_circuit = tfq.convert_to_tensor([cirq.Circuit()])
        self.computation_layer = tfq.layers.ControlledPQC(self.circuit, self.observables)        


    # Creates the circuit of VQC model for the qubits used as input and a time horizon T, for L layers
    def createCircuit(self, qubits, L, T):

        n_qubits= len(qubits)

        # Inputs
        nInputs= T
        inSym = sympy.symbols(f'x(0:{nInputs})')
        inSym = np.asarray(inSym).reshape(nInputs)

        # Weights
        WSym = sympy.symbols(f'W(0:{ L })'+f'_(0:{n_qubits})'+f'_(0:{3})')
        WSym = np.asarray(WSym).reshape(( L,  n_qubits, 3))

        # Circuit
        circuit = cirq.Circuit()
        circuit += cirq.Circuit(cirq.rx(inSym[i])(self.qubits[i]) for i in range(nInputs))

        for l, WSym_l in enumerate(WSym):

            circuit += cirq.Circuit([cirq.rx(WSym_l[i, 0])(q), cirq.ry(WSym_l[i, 1])(q), cirq.rz(WSym_l[i, 2])(q) ] for i, q in enumerate(qubits))
            cz_ops = []
            for qi in range( n_qubits):
                cz_ops.append(cirq.CZ(qubits[qi], qubits[(qi+1)%n_qubits]))
            circuit+= cirq.Circuit(cz_ops)

        return circuit, list(inSym.flat), list(WSym.flat)


    # Execution of this layer for the given inputs=[x(t-1), x(t-2), ..., x(t-T)]
    def call(self, inputs):
        
        # get the number of input patterns, and prepare the batch for parallel execution
        batch_dim = tf.gather(tf.shape(inputs[0]), 0)
        tiled_up_circuits = tf.repeat(self.empty_circuit, repeats=batch_dim)
        tiled_up_W = tf.tile(self.W, multiples=[batch_dim, 1])
        tiled_up_inputs = tf.tile(inputs[0], multiples=[1, 1])
        
        # If autoScaleInputs is True, then weight inputs as Wi*x(t-i)
        if self.Wi is not None:
            scaled_inputs = tf.einsum("i,ji->ji", self.Wi, tiled_up_inputs)
        else:
            scaled_inputs= tiled_up_inputs
        
        # Get all parameters and inputs in the sorted symbol order
        joined_vars = tf.concat([tiled_up_W, scaled_inputs], axis=1)
        joined_vars = tf.gather(joined_vars, self.indices, axis=1)
        
        # Then execute the circuit for the inputted batch
        return self.computation_layer([tiled_up_circuits, joined_vars])




###################################################################
# QVCNetwork neural network model
###################################################################
class VQCNetwork(tf.keras.Model):
    
    # Constructor
    # Inputs:
    #   qubits: A list containing T qubits
    #   L: Number of layers of the variational quantum circuit
    #   autoScaleInputs: True to weight input values; false to provide raw inputs to the quantum circuit
    #   autoScaleOutputs: True to scale and bias circuit output values; false to provide raw circuit outputs as results
    def __init__(self, qubits, L, autoScaleInputs= False, autoScaleOutputs=False):
        
        super(VQCNetwork, self).__init__()

        # set parameters
        self.qubits= qubits
        self.T= len(qubits)
        self.autoScaleInputs= autoScaleInputs
        self.autoScaleOutputs= autoScaleOutputs
        self.n_outputs= 1
        
        # Create quantum circuit and measurement
        self.input_tensor = tf.keras.Input(shape=(self.T, ), dtype=tf.dtypes.float32, name='inputTensor')
        self.quantumLayer = QuantumLayer(qubits, self.T, autoScaleInputs=autoScaleInputs)([self.input_tensor])
        process = tf.keras.Sequential([tf.keras.layers.Lambda(lambda x: x)])
        output = process(self.quantumLayer)
        self.net= tf.keras.Model(inputs=[self.input_tensor], outputs=output)
        
        # Set output scale and bias parameters
        if self.autoScaleOutputs:
            Wo_init = tf.ones(shape=(1, self.n_outputs))
            self.Wo = tf.Variable(initial_value=Wo_init, dtype="float32", trainable=True, name="Woutputs")
            Bo_init = tf.zeros(shape=(1, self.n_outputs))
            self.Bo = tf.Variable(initial_value=Bo_init, dtype="float32", trainable=True, name="Boutputs")
        else:
            self.Wo= None
            self.Bo= None

    # Forward pass
    def call(self, x):

        out= self.net(x)
        
        if self.autoScaleOutputs:
            repBo= tf.repeat(self.Bo,repeats=tf.shape(out)[0],axis=0)
            repWo= tf.repeat(self.Wo,repeats=tf.shape(out)[0],axis=0)
            out_scaled= tf.math.add(tf.math.multiply(out, repWo) , repBo)
        else:
            out_scaled= out
        return out_scaled
 

for zone in range(1, 4):


    XTr, YTr, XTs, YTs= LoadData(T=7, zone=zone)
    
    T= XTr.shape[1]
    n_qubits= T
    qubits = cirq.GridQubit.rect(1, n_qubits)
    
    # Quantum Layers
    L= 1
    
    resultsFile= 'QNNresZ{}.pkl'.format(zone) # None for not saving results data
    lr= 0.01 # Learning rate of training algorithm
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
    
        # VQC model definition
        model= VQCNetwork(qubits, L, autoScaleInputs= True, 
                          autoScaleOutputs=True)
    
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
    
    
    
