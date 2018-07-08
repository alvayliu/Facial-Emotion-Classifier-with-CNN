## Code inspired by Christian Perone, http://blog.christianperone.com/

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import six.moves.cPickle as pickle
import os
import gzip
import numpy as np
import theano
import lasagne
import csv
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import visualize
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix



def load_dataset():
	#Read csv file
	data = pd.read_csv('fer2013.csv')

	#Number of samples
	n_samples = len(data)
	n_samples_train = 28709
	n_samples_test = 3589
	n_samples_validation = 3589

	#Pixel width and height
	w = 48
	h = 48

	#Separating labels and features respectively
	y = np.zeros(n_samples)
	X = np.zeros((n_samples, 1, w, h))
	for i in range(n_samples):
	    X[i] = np.fromstring(data['pixels'][i], dtype=int, sep=' ').reshape((-1, 1, w, h))
	    y[i] = data['emotion'][i]

	y = y.astype(np.int32)

	#Training set   
	X_train = X[:n_samples_train]
	y_train = y[:n_samples_train]
	#Testing set
	X_test = X[n_samples_train : (n_samples_train + n_samples_test)]  
	y_test = y[n_samples_train : (n_samples_train + n_samples_test)]
	#Validation set
	X_val = X[(n_samples_train + n_samples_test):]  
	y_val = y[(n_samples_train + n_samples_test):]
	return X_train, y_train, X_val, y_val, X_test, y_test


X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()

net1 = NeuralNet(
	layers=[('input', layers.InputLayer),
			('conv2d1', layers.Conv2DLayer),
			('conv2d2', layers.Conv2DLayer),
			('conv2d3', layers.Conv2DLayer),
			('maxpool1', layers.MaxPool2DLayer),
			('conv2d4', layers.Conv2DLayer),
			('conv2d5', layers.Conv2DLayer),
			('conv2d6', layers.Conv2DLayer),
			('maxpool2', layers.MaxPool2DLayer),
			('conv2d7', layers.Conv2DLayer),
			('conv2d8', layers.Conv2DLayer),
			('conv2d9', layers.Conv2DLayer),
			('maxpool3', layers.MaxPool2DLayer),
			('dropout1', layers.DropoutLayer),
			('dense1', layers.DenseLayer),
			('dropout2', layers.DropoutLayer),
			('dense2', layers.DenseLayer),
			('output', layers.DenseLayer),
			],
	
	# input layer
	input_shape=(None, 1, 48, 48),
	
	# layer conv2d1
	conv2d1_num_filters=32,
	conv2d1_filter_size=(3, 3),
	conv2d1_pad='same',
	conv2d1_nonlinearity=lasagne.nonlinearities.rectify, #Rectify -> ReLu 
	conv2d1_W=lasagne.init.GlorotUniform(),   #intitialize weights uniformly: Xavier initiation  
	
	# layer conv2d2
	conv2d2_num_filters=32,
	conv2d2_filter_size=(3, 3),
	conv2d2_pad='same',
	conv2d2_nonlinearity=lasagne.nonlinearities.rectify, #Rectify -> Relu
	
	# layer conv3d3
	conv2d3_num_filters=32,
	conv2d3_filter_size=(3, 3),
	conv2d3_pad='same',
	conv2d3_nonlinearity=lasagne.nonlinearities.rectify, #Rectify -> Relu
	
	# layer maxpool3
	maxpool1_pool_size=(2, 2),

	# layer conv3d4
	conv2d4_num_filters=64,
	conv2d4_filter_size=(3, 3),
	conv2d4_pad='same',
	conv2d4_nonlinearity=lasagne.nonlinearities.rectify, #Rectify -> Relu

	# layer conv3d5
	conv2d5_num_filters=64,
	conv2d5_filter_size=(3, 3),
	conv2d5_pad='same',
	conv2d5_nonlinearity=lasagne.nonlinearities.rectify, #Rectify -> Relu

	# layer conv3d6
	conv2d6_num_filters=64,
	conv2d6_filter_size=(3, 3),
	conv2d6_pad='same',
	conv2d6_nonlinearity=lasagne.nonlinearities.rectify, #Rectify -> Relu

	maxpool2_pool_size=(2, 2),

	# layer conv3d7
	conv2d7_num_filters=128,
	conv2d7_filter_size=(3, 3),
	conv2d7_pad='same',
	conv2d7_nonlinearity=lasagne.nonlinearities.rectify, #Rectify -> Relu

	# layer conv3d8
	conv2d8_num_filters=128,
	conv2d8_filter_size=(3, 3),
	conv2d8_pad='same',
	conv2d8_nonlinearity=lasagne.nonlinearities.rectify, #Rectify -> Relu

	# layer conv3d9
	conv2d9_num_filters=128,
	conv2d9_filter_size=(3, 3),
	conv2d9_pad='same',
	conv2d9_nonlinearity=lasagne.nonlinearities.rectify, #Rectify -> Relu

	# layer maxpool1
	maxpool3_pool_size=(2, 2),  
	
	# dropout1
	dropout1_p=0.2,    
	
	# dense
	dense1_num_units=64,
	dense1_nonlinearity=lasagne.nonlinearities.rectify, #Relu 
	
	# dropout2
	dropout2_p=0.2,   

	# dense
	dense2_num_units=64,
	dense2_nonlinearity=lasagne.nonlinearities.rectify, #Relu

	# output
	output_nonlinearity=lasagne.nonlinearities.softmax,
	output_num_units=7,
	
	# optimization method params
	update=nesterov_momentum,
	update_learning_rate=0.005,
	update_momentum=0.9,
	max_epochs=12,
	verbose=1,
	)


# Train the network
nn = net1.fit(X_train, y_train)

# Predict classes
preds = net1.predict(X_test)

# Calculate test accuracy, and accuracy for each class. 
R = np.zeros(7) 
T = np.zeros(7)
R_all = 0
T_all = len(y_test)

for i in range(T_all):
	T[y_test[i]] += 1
	if y_test[i]==preds[i]:
		R_all += 1
		R[y_test[i]] += 1

print("Test accuracy: ", R_all/T_all)
print("Accuracy Angry: ", R[0]/T[0])
print("Accuracy Disgust: ", R[1]/T[1])
print("Accuracy Fear: ", R[2]/T[2])
print("Accuracy Happy: ", R[3]/T[3])
print("Accuracy Sad: ", R[4]/T[4])
print("Accuracy Surprise: ", R[5]/T[5])
print("Accuracy Neutral: ", R[6]/T[6])


