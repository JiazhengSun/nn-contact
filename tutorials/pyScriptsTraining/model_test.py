from __future__ import print_function
import scipy.io as sio
import numpy as np
from keras.models import load_model
import csv
import os

mpath = '/Users/jiazhengsun/Desktop/nn-contact/data/NN-contact-force/keras_model/3D/'
model = load_model(os.path.join(mpath,'3D-rect-C3.h5'))

x_test_1 = [[1, 0, 0, 0, 1, 0, 0, 0, 1,
0, 0, 0, 0.285714, -0.28, 0.285714, 0.285714, -0.28,
0.285714, 0, 0.0988, 0]]

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

#Line data
# 1, 0, 0, 0, 0.930141, 0.367203, 0,
# -0.367203, 0.930141, 0.0666667, 0, 0, 0.285714, -0.146505, 
# 0.365241, 0.285714, -0.274792, 0.250401, 0, 0.243477, -0.0924968  

#keras output => 1.6008365e-01 8.3991617e-01 1.6176570e-07
#tinydnn output => 0.160084 0.839916 1.61766e-07 


#Point data
# 0.997834, -0.0225904, -0.0617758, 0.0347001, 0.978643, 0.20262, 0.0558792, 
# -0.204325, 0.977307, 0.0441396, -0.178487, -0.21877, 0.17429, -0.111783, 
# 0.113199, 0.0276258, -0.30103, 0.147426, -0.895265, 0.232987, -0.162503  

#keras output => 9.8978257e-01 1.0215883e-02 1.5291808e-06
#tinydnn output => 0.989783 0.0102159 1.52918e-06  


#Surface data
# 1, 0, 0, 0, 1, 0, 0, 0, 1,
# 0, 0, 0, 0.285714, -0.28, 0.285714, 0.285714, -0.28,
# 0.285714, 0, 0.0988, 0

#keras output => 9.9997783e-01 9.8859516e-09 2.2188808e-05
#tinydnn output => 0.999978 9.88593e-09 2.21888e-05  
