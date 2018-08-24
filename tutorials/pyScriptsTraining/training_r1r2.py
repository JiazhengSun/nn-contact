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

x_temp = []
y_temp = []
list_size = 0
with open('/home/jsun303/Desktop/dart-NN-contact-force/data/NN-contact-force/train_data/poly_r2_sym.csv') as csvfile:
	raw_data = csv.reader(csvfile, delimiter='\n')
	for row in raw_data:
		data_string = row[0]
		data_digit_str = data_string.split(',')
		list_size = len(data_digit_str)
		x_temp.append(data_digit_str[:list_size - 2])
		y_temp.append(data_digit_str[list_size - 2])

x_total = []
y_total = []
for i in range(len(x_temp)):
	y_total.append(float(y_temp[i]))
	indi_data_set = []
	for item in x_temp[i]:
		#x_total.append(float(item))
		indi_data_set.append(float(item))
	x_total.append(indi_data_set)

y_total = np.transpose(y_total)

length = y_total.shape[0]

#x_total: 5 elements per array, NUMSAM number of arrays. So dim = 5.
#y_total: 1 array, NUMSAM labels. To make it match x_total, need to transpose it.

x_train = x_total[:int(length * 0.8)]
x_test = x_total[int(length*0.8):]
y_train = y_total[:int(length* 0.8)]/100.0
y_test = y_total[int(length*0.8):]/100.0

x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test) 


model = Sequential()
model.add(Dense(64, input_dim=5, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

decay = 0.02/200
adam = optimizers.adam(lr=0.02, decay=decay)
model.compile(loss='mean_squared_error', optimizer=adam)
#print(model.metrics_names)

history = model.fit(x_train, y_train, validation_split = 0.33, epochs=200, batch_size=100)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'],loc='upper left')
plt.show()
score = model.evaluate(x_test, y_test, batch_size=100)
print(score)

#print(y_test)
#y_pred = model.predict(x_test, batch_size=100)
#print(y_pred)
#print(np.linalg.norm(y_pred-y_test)/np.linalg.norm(y_test))
#
#model.save('pdd-r2-feb24-2pi.h5')
model.save('poly_r2_sym.h5')