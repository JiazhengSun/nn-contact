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
from random import shuffle
import os

save_model_path = "/Users/jiazhengsun/Desktop/nn-contact/data/NN-contact-force/keras_model/3D/"

x_temp = []
y_temp = []
list_size = 0
with open('/Users/jiazhengsun/Desktop/nn-contact/data/NN-contact-force/train_data/3D/rect_r3_3D_sym.csv') as csvfile:
	raw_data = csv.reader(csvfile, delimiter='\n')

	shuffle_list = []
	for indi_data in raw_data:
		shuffle_list.append(indi_data)
	shuffle(shuffle_list)
	
	for row in shuffle_list:
		data_string = row[0]
		data_digit_str = data_string.split(',')
		list_size = len(data_digit_str)
		x_temp.append(data_digit_str[:list_size - 4])
		y_temp.append(data_digit_str[list_size - 4:list_size-1])

x_total = []
y_total = []
for i in range(len(x_temp)):
	indi_data_set = []
	for item in x_temp[i]:
		indi_data_set.append(float(item))
	x_total.append(indi_data_set)

for j in range(len(y_temp)):
	indi_data_set = []
	for item in y_temp[j]:
		indi_data_set.append(float(item))
	y_total.append(indi_data_set)

# y_total = np.transpose(y_total)
# print(y_total)
y_total = np.array(y_total)
length = y_total.shape[0]

# Scale tau_y by 45 to make its average matches the range of that of fx and fz
for eachSet in y_total:
	eachSet[-1] *= 45.0;

#x_total: 12 elements per array, NUMSAM number of arrays. So dim = 12.
#y_total: 1 array, NUMSAM labels. To make it match x_total, need to transpose it.

x_test = x_total[int(length*0.8):]
x_test = np.array(x_test)
y_test = y_total[int(length*0.8):]/100.0
y_test = np.array(y_test) 


x_train = np.array(x_total)
y_train = y_total[:]/100.0

model = Sequential()
model.add(Dense(256, input_dim=9, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(3))

decay = 0.02/200
adam = optimizers.adam(lr=0.02, decay=decay)
model.compile(loss='mean_squared_error', optimizer=adam)
#print(model.metrics_names)

history = model.fit(x_train, y_train, validation_split = 0.2, epochs=200, batch_size=100)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'],loc='upper left')
plt.show()
# score = model.evaluate(x_test, y_test, batch_size=100)
# print(score)

print(y_test)
y_pred = model.predict(x_test, batch_size=100)
print(y_pred)
print("px testing: ")
print(np.linalg.norm(y_pred[:,0]-y_test[:,0])/np.linalg.norm(y_test[:,0]))
print("pz testing: ")
print(np.linalg.norm(y_pred[:,1]-y_test[:,1])/np.linalg.norm(y_test[:,1]))
print("ty testing: ")
print(np.linalg.norm(y_pred[:,2]-y_test[:,2])/np.linalg.norm(y_test[:,2]))

save_name = 'rect_r3_3D_sym.h5' 
model.save(os.path.join(save_model_path, save_name))





# Methods that are used for help

# 1. Compare avg between linear impulse and angular impulse, to determine the scale for matching
# totalPx = 0
# totalPz = 0
# totalTy = 0

# for eachSet in y_total:
# 	totalPx += abs(eachSet[0])
# 	totalPz += abs(eachSet[1])
# 	totalTy += abs(eachSet[2])

# avgPx = totalPx/len(y_total)
# avgPz = totalPz/len(y_total)
# avgTy = totalTy/len(y_total)
# print("Length of total data is: ", len(y_total))
# print("Average of px is ", avgPx)
# print("Average of pz is ", avgPz)
# print("Average of ty is ", avgTy)

# ('Length of total data is: ', 500000)
# ('Average of px is ', 660.3843540530806)
# ('Average of pz is ', 661.0706745713851)
# ('Average of ty is ', 14.747050401035883)


