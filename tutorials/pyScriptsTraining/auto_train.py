from keras.models import Sequential
from keras.layers import Dense
from keras import utils
from keras import optimizers
import scipy.io as sio
import numpy as np
from sklearn import preprocessing
import csv
from keras import backend as K
import matplotlib.pyplot as plt
import os
import networkHelpers

train_data_path = "/home/jsun303/Desktop/dart-NN-contact-force/data/NN-contact-force/train_data/3D/"
save_model_path = "/home/jsun303/Desktop/dart-NN-contact-force/data/NN-contact-force/keras_model/3D/"
os.chdir(train_data_path)

for filename in os.listdir(train_data_path):
	isClassifier = False
	with open(filename, 'rb') as csvFile:
		if "c1" in filename or "c2" in filename or "c3" in filename:
			print("Classifier model training!!")
			isClassifier = True
			data = networkHelpers.extractData(csvFile, filename, isClassifier)
			networkHelpers.classifyTrain(data[0], data[1], data[2], data[3], save_model_path, filename)
		elif "r1" in filename or "r2" in filename or "r3" in filename:
			print("Regression model training!")
			data = networkHelpers.extractData(csvFile, filename, isClassifier)
			networkHelpers.regressTrain(data[0], data[1], data[2], data[3], save_model_path, filename)