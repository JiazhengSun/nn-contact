from __future__ import print_function
import scipy.io as sio
import numpy as np
from keras.models import load_model
import csv
import os
#Only comma, no \n in one cell
#Use \n to seperate each layer

#Convention for python and c++ file IO communication!!!!!:
#\n separate between layers. 
#Comma separate between each neuron. 
#Space separate within neuron set

mpath = '/Users/jiazhengsun/Desktop/nn-contact/data/NN-contact-force/keras_model/3D/'
wpath = '/Users/jiazhengsun/Desktop/nn-contact/data/NN-contact-force/weights_csv/3D/'
bpath = '/Users/jiazhengsun/Desktop/nn-contact/data/NN-contact-force/bias_csv/3D/'
model = load_model(os.path.join(mpath,'rect_r3_3D_sym.h5'))

with open(wpath + 'rect_r3_sym_3D_weights.csv','wb') as wf:
    #Classfier Weights : first layer: 5*32, second layer: 32*32, third layer: 32*16
    #Regression Weights: frist layer:: 5*64, second layer: 64* 64, third layer: 64*32
    for i in range(4):
        indi_str = ""
        total_str = ""
        #Wname = 'W' + str(i)
        #print(Wname)
        myWeights = model.layers[i].get_weights()[0]
        myWeights = myWeights.tolist()
        #print("layer",i)
        for lst in myWeights:
            indi_str = ' '.join(str(w) for w in lst) #Now the wieghts for each input to layer doens't have comma
            #print(indi_str + '\n')
            total_str += indi_str
            total_str += ","
            #print(total_str)
        writer = csv.writer(wf)
        writer.writerow([total_str])


with open(bpath + 'rect_r3_sym_3D_bias.csv','wb') as bf:
    for i in range(4):
        #Bname = 'B' + str(i)
        indi_str = ""
        total_str = ""
        myBias = model.layers[i].get_weights()[1]
        myBias = np.transpose(myBias)
        myBias = myBias.tolist()
        writer = csv.writer(bf)
        writer.writerow([myBias])

wf.close()