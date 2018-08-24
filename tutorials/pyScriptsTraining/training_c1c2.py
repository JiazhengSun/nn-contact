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
with open('/Users/jiazhengsun/Desktop/nn-contact/data/NN-contact-force/train_data/rect_c2_sym.csv') as csvfile:
	raw_data = csv.reader(csvfile, delimiter='\n')
	for row in raw_data:
		data_string = row[0]
		data_digit_str = data_string.split(',')
		list_size = len(data_digit_str)
		#print(data_digit_str[list_size - 2])
		x_temp.append(data_digit_str[:list_size - 2])
		y_temp.append(data_digit_str[list_size - 2])

x_total = []
y_total = []
for i in range(len(x_temp)):
	y_total.append(int(y_temp[i]))
	indi_data_set = []
	for item in x_temp[i]:
		#x_total.append(float(item))
		indi_data_set.append(float(item))
	x_total.append(indi_data_set)


y_total = np.transpose(y_total)
one_hot_labels = utils.to_categorical(y_total, num_classes=3)

length = y_total.shape[0]

#x_total: 5 elements per array, NUMSAM number of arrays. So dim = 5.
#y_total: 1 array, NUMSAM labels. To make it match x_total, need to transpose it.

x_train = x_total[:int(length * 0.8)]
x_test = x_total[int(length*0.8):]
y_train = one_hot_labels[:int(length* 0.8)]
y_test = one_hot_labels[int(length*0.8):]

x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

model = Sequential()
model.add(Dense(32, input_dim=5, activation='tanh'))
model.add(Dense(32, activation='tanh'))
model.add(Dense(16, activation='tanh'))
model.add(Dense(3, activation='softmax'))

# decay = 0.002/50
# adam = optimizers.Adam(lr=0.002, decay=decay)
# model.compile(loss='mean_squared_error', optimizer=adam)
# print(model.metrics_names)

# adam = optimizers.rmsprop(lr=0.005)

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train, y_train, validation_split = 0.33, epochs=200, batch_size=100)
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'],loc='upper left')
# plt.show()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'],loc='upper left')
plt.show()


# score = model.evaluate(x_test, y_test, batch_size=100)
# print(score)
# print(model.predict(x_test, batch_size=100))

#model.save('unsym/c2-unsym-feb26.h5')
model.save('rect_c2_sym.h5')
