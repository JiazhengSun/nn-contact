from __future__ import print_function
import scipy.io as sio
import numpy as np
from keras.models import load_model
import csv
import os

mpath = '/Users/jiazhengsun/Desktop/nn-contact/data/NN-contact-force/keras_model/Unsym/'
model = load_model(os.path.join(mpath,'3D-rect-unsym-C3.h5'))

x_test_1 = [[0.997834, -0.0225904, -0.0617758, 0.0347001, 0.978643, 0.20262, 0.0558792, 
-0.204325, 0.977307, 0.0441396, -0.178487, -0.21877, 0.17429, -0.111783, 
0.113199, 0.0276258, -0.30103, 0.147426, -0.895265, 0.232987, -0.162503  ]]

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


#Point data
# 0.997834, -0.0225904, -0.0617758, 0.0347001, 0.978643, 0.20262, 0.0558792, 
# -0.204325, 0.977307, 0.0441396, -0.178487, -0.21877, 0.17429, -0.111783, 
# 0.113199, 0.0276258, -0.30103, 0.147426, -0.895265, 0.232987, -0.162503  

#Line data
# 1, 0, 0, 0, 0.930141, 0.367203, 0,
# -0.367203, 0.930141, 0.0666667, 0, 0, 0.285714, -0.146505, 
# 0.365241, 0.285714, -0.274792, 0.250401, 0, 0.243477, -0.0924968  

#Surface data
# 1, 0, 0, 0, 1, 0, 0, 0, 1,
# 0, 0, 0, 0.285714, -0.28, 0.285714, 0.285714, -0.28,
# 0.285714, 0, 0.0988, 0

#[[-1.6367627 -1.3376367]]				-1.63676 -1.33764
#[[-1.6106048 -1.7932341  3.6087656]]	-1.6106 -1.79323 3.60877 
#[[-4.678535  -5.4778123  2.8040876]]	-4.67854 -5.47781 2.80409 

#[[9.9999881e-01 1.2305836e-06 5.1843855e-08]] 0.999999 1.23058e-06 5.18439e-08 
#[[1.0000000e+00 1.6036619e-12 1.9720470e-10]] 1 1.60366e-12 1.97205e-10 
#[[9.0215653e-01 9.7837240e-02 6.2913100e-06]] 0.902157 0.0978372 6.2913e-06 