from __future__ import print_function
import scipy.io as sio
import numpy as np
from keras.models import load_model
import csv
import os

mpath = '/home/jsun303/Desktop/dart-NN-contact-force/data/NN-contact-force/keras_model'
model = load_model(os.path.join(mpath,'rect_r2_sym.h5'))

x_test_1 = [[0.510521,	-0.859865,	-2.08544,	-6.72442,	-2.32748], 
			[-0.328948,	-0.944348,	-0.0272732,	9.12826,	-4.19388],
			[0.16081,	0.986985,	3.58803,	-9.08075,	-0.740867],
			[-0.132366,	-0.991201,	2.15397,	7.04784,	0.526174],
			[0.272638,	-0.962117,	3.54762,	7.57354,	-4.66197]]

x_test_1 = np.array(x_test_1)
print(model.predict(x_test_1, batch_size=1))

# for i in range(4):
# 	Wname = 'W' + str(i)
# 	Bname = 'B' + str(i)
# 	Wmat = model.layers[i].get_weights()[0]
# 	#Wmat = np.transpose(Wmat)
# 	print(Wname)
# 	for i in range(len(Wmat)):
# 		print(Wmat[i])
# 	#Bmat = model.layers[i].get_weights()[1]
# 	#Bmat = np.transpose(Bmat)
# 	#print(Bmat.shape)

#c1
# [[1.0000000e+00 1.4670566e-09 5.0629717e-12]
#  [2.4774924e-03 2.0718331e-09 9.9752253e-01]
#  [5.5542000e-07 1.5256164e-12 9.9999940e-01]
#  [9.9987102e-01 1.2863059e-04 4.0769757e-07]
#  [6.0155594e-07 9.9999940e-01 1.3631557e-15]]


#c2
# [[1.0000000e+00 3.2690248e-12 1.0985155e-15]
#  [1.0000000e+00 6.4753428e-12 5.9408507e-12]
#  [2.1679217e-12 2.4023355e-16 1.0000000e+00]
#  [1.2339571e-11 1.0000000e+00 1.7224443e-20]
#  [1.0000000e+00 7.4202062e-09 5.1415171e-15]]

# r1
# [[ 9.58168  ]
#  [-8.49335  ]
#  [ 6.777815 ]
#  [-2.6062138]
#  [-3.2998772]]
