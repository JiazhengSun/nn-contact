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

def extractData(myFile, filename, flag):
	x_temp = []
	y_temp = []
	raw_data = csv.reader(myFile, delimiter='\n')
	for row in raw_data:
		data_string = row[0]
		data_digit_str = data_string.split(',')
		list_size = len(data_digit_str)
		if flag == True:
			x_temp.append(data_digit_str[:list_size - 2])
			y_temp.append(data_digit_str[list_size - 2])
		else:
			x_temp.append(data_digit_str[:list_size - 4])
			y_temp.append(data_digit_str[list_size - 4 : list_size - 1])			

	x_total = []
	for i in range(len(x_temp)):
		indi_data_set = []
		for item in x_temp[i]:
			indi_data_set.append(float(item))
		x_total.append(indi_data_set)


	y_total = []
	if flag == True: #is classfier, so 3 classes of label
		for i in range(len(x_temp)):
			y_total.append(int(y_temp[i]))
		y_total = np.transpose(y_total)
		one_hot_labels = utils.to_categorical(y_total, num_classes=3)
	else:
		for j in range(len(y_temp)):
			indi_data_set = []
			for item in y_temp[j]:
				indi_data_set.append(float(item))
			y_total.append(indi_data_set)
		y_total = np.array(y_total)

	xtrain = []
	ytrain = []
	xtest = []
	ytest = []
	length = y_total.shape[0]
	x_train = x_total[:int(length * 0.8)]
	x_test = x_total[int(length*0.8):]

	if flag == True:
		y_train = one_hot_labels[:int(length* 0.8)]
		y_test = one_hot_labels[int(length*0.8):]
	else:
		y_train = y_total[:int(length* 0.8)]/100.0
		y_test = y_total[int(length*0.8):]/100.0
	
	x_train = np.array(x_train)
	x_test = np.array(x_test)
	y_train = np.array(y_train)
	y_test = np.array(y_test) 

	return (x_train, y_train, x_test, y_test)

def classifyTrain(x_train, y_train, x_test, y_test, save_model_path, filename):
	model = Sequential()
	model.add(Dense(32, input_dim=12, activation='tanh'))
	model.add(Dense(32, activation='tanh'))
	model.add(Dense(16, activation='tanh'))
	model.add(Dense(3, activation='softmax'))
	model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
	# Graphing and printing out the result

	history = model.fit(x_train, y_train, validation_split = 0.33, epochs=200, batch_size=100)
	# plt.plot(history.history['acc'])
	# plt.plot(history.history['val_acc'])
	# plt.title('model accuracy')
	# plt.ylabel('accuracy')
	# plt.xlabel('epoch')
	# plt.legend(['train', 'test'],loc='upper left')
	# plt.show()

	# score = model.evaluate(x_test, y_test, batch_size=100)
	# print(score)
	# print(model.predict(x_test, batch_size=100))

	#save model based on name
	save_name = ""
	if "c1" in filename:
		if "unsym" in filename:
			if "poly" in filename:
				save_name = "poly_c1_unsym.h5"
			else:
				save_name = "rect_c1_unsym.h5"
		else:
			if "poly" in filename:
				save_name = "poly_c1_sym.h5"
			else:
				save_name = "rect_c1_sym.h5"
	elif "c2" in filename:
		if "unsym" in filename:
			if "poly" in filename:
				save_name = "poly_c2_unsym.h5"
			else:
				save_name = "rect_c2_unsym.h5"
		else:
			if "poly" in filename:
				save_name = "poly_c2_sym.h5"
			else:
				save_name = "rect_c2_sym.h5"
	else:
		if "unsym" in filename:
			if "poly" in filename:
				save_name = "poly_c3_unsym.h5"
			else:
				save_name = "rect_c3_unsym.h5"
		else:
			if "poly" in filename:
				save_name = "poly_c3_sym.h5"
			else:
				save_name = "rect_c3_sym.h5"

	model.save(os.path.join(save_model_path, save_name))



def regressTrain(x_train, y_train, x_test, y_test, save_model_path,filename):
	model = Sequential()
	model.add(Dense(64, input_dim=12, activation='relu'))
	model.add(Dense(64, activation='relu'))
	model.add(Dense(32, activation='relu'))
	model.add(Dense(3))

	decay = 0.02/200
	adam = optimizers.adam(lr=0.02, decay=decay)
	model.compile(loss='mean_squared_error', optimizer=adam)
	#print(model.metrics_names)

	history = model.fit(x_train, y_train, validation_split = 0.33, epochs=200, batch_size=100)
	# plt.plot(history.history['loss'])
	# plt.plot(history.history['val_loss'])
	# plt.title('model loss')
	# plt.ylabel('loss')
	# plt.xlabel('epoch')
	# plt.legend(['train', 'test'],loc='upper left')
	# plt.show()
	# score = model.evaluate(x_test, y_test, batch_size=100)
	# print(score)
	save_name = ""
	if "r1" in filename:
		if "unsym" in filename:
			if "poly" in filename:
				save_name = "poly_r1_unsym.h5"
			else:
				save_name = "rect_r1_unsym.h5"
		else:
			if "poly" in filename:
				save_name = "poly_r1_sym.h5"
			else:
				save_name = "rect_r1_sym.h5"
	elif "r2" in filename:
		if "unsym" in filename:
			if "poly" in filename:
				save_name = "poly_r2_unsym.h5"
			else:
				save_name = "rect_r2_unsym.h5"
		else:
			if "poly" in filename:
				save_name = "poly_r2_sym.h5"
			else:
				save_name = "rect_r2_sym.h5"
	else:
		if "unsym" in filename:
			if "poly" in filename:
				save_name = "poly_r3_unsym.h5"
			else:
				save_name = "rect_r3_unsym.h5"
		else:
			if "poly" in filename:
				save_name = "poly_r3_sym.h5"
			else:
				save_name = "rect_r3_sym.h5"		
	model.save(os.path.join(save_model_path, save_name))
