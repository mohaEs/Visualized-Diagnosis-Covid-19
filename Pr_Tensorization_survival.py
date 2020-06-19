# -*- coding: utf-8 -*-
"""
Created on Wed May  1 10:28:19 2019

@author: User
"""

# Multilayer Perceptron
import pandas
import numpy
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# from tensorflow import set_random_seed
# set_random_seed(2)

import tensorflow.keras
import math
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2DTranspose,Input, Reshape, Conv2D, Flatten
from tensorflow.keras.layers import Dense,concatenate
from sklearn.metrics import mean_squared_error
import argparse
#from tensorflow.keras.utils.np_utils import to_categorical
import tensorflow as tf
import matplotlib.pyplot as plt


from tensorflow.keras.layers import Dropout
import numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from scipy.stats import pearsonr, spearmanr

####################
####### Functions #############
####################


def custom_loss_2 (y_true, y_pred):
    
    Red_true=y_true[:,:,:,0]
    Red_pred=y_pred[:,:,:,0]
    
    blue_true=y_true[:,:,:,2]
    blue_pred=y_pred[:,:,:,2]
    
    gr_true=y_true[:,:,:,1]
    gr_pred=y_pred[:,:,:,1]
    
    A_red= tensorflow.keras.losses.mean_absolute_error(Red_true, Red_pred)
    A_gr= tensorflow.keras.losses.mean_absolute_error(gr_pred, gr_true)
    A_blue= tensorflow.keras.losses.mean_absolute_error(blue_true, blue_pred)
    
    A=(5*A_red)+A_blue+A_gr
    #A = tensorflow.keras.losses.mean_absolute_error(y_true, y_pred)
    #B = keras.losses.categorical_crossentropy(y_true[:,-4:], y_pred[:,-4:])
    return A





def lrelu(x): #from pix2pix code
    a=0.2
    # adding these together creates the leak part and linear part
    # then cancels them out by subtracting/adding an absolute value term
    # leak: a*x/2 - a*abs(x)/2
    # linear: x/2 + abs(x)/2

    # this block looks like it has 2 inputs on the graph unless we do this
    x = tf.identity(x)
    return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)

def lrelu_output_shape(input_shape):
    shape = list(input_shape)
    return tuple(shape)
from tensorflow.keras.layers import Lambda
layer_lrelu=Lambda(lrelu, output_shape=lrelu_output_shape)

def my_unit_circle(r):
#    d = 2*r + 1
#    rx, ry = d/2, d/2
#    x, y = np.indices((d, d))
#    h= (np.abs(np.hypot(rx - x, ry - y)-r) < 0.5).astype(int)
    
    xx, yy = np.mgrid[:45, :45]
    circle_inds = (xx - 22) ** 2 + (yy - 22) ** 2
    circle_bool = circle_inds < (r)
    
    circle=1.0*circle_bool
    # 45,45: 968
    # 0,0: 968
    # 500 is biggest circle
    # plt.imshow(circle)
    
    return circle


def FnCreateTargetImages(Labels):
    # Neon color
    OutputImages=np.zeros(shape=(len(Labels),45,45,3))    
    
    r=400
    CIRCLE=my_unit_circle(r)
    # plt.imshow(CIRCLE)
    
    for i in range(len(Labels)):
          
        if Labels[i]==0:

            OutputImages[i,:,:,1]=CIRCLE
            
        elif Labels[i]==1:
            OutputImages[i,:,:,0]=CIRCLE
            OutputImages[i,:,:,1]=CIRCLE
            
        elif Labels[i]==2:
            OutputImages[i,:,:,0]=CIRCLE
            
#        elif Labels[i]==2:
#            OutputImages[i,6:17,6:17,0]=0.6
#            OutputImages[i,6:17,6:17,1]=0.6
#            OutputImages[i,6:17,6:17,2]=0.6
#            
#            OutputImages[i,3:20,3,:]=0.6
#            OutputImages[i,3:20,20,:]=0.6
#            OutputImages[i,20,3:20,:]=0.6
#            OutputImages[i,3,3:20,:]=0.6
            

                    
            # plt.figure()
            # plt.imshow(OutputImages[3,:,:,:])
    return OutputImages



def FnCreateValidLabes(Labels):
    return range(len(Labels))

     

####################
###### End of functions ##############    
####################    


####################
###### Reading input arguments ##############        
#########################################################
######### Hyper paramters configurations ##################
########### ##########################################################

#
#i=64
#j=16
#k=32
#n=105

n_splits=10

parser = argparse.ArgumentParser()
parser.add_argument("--output", )
parser.add_argument("--max_epochs",  )
parser.add_argument("--BatchSize", )
parser.add_argument("--k", )
parser.add_argument("--m", )
a = parser.parse_args()


a.max_epochs=100
a.BatchSize=10
a.output='./1/'

import os
try:
    os.stat(a.output)
except:
    os.mkdir(a.output) 

##
#i=int(a.i)
#j=int(a.j)
#k=int(a.k)



####################
###### Reading Data ##############    
####################    




Dataset_covid = pandas.read_csv('./Data_cleaned_survival.csv', low_memory=False)
# creating independent features X and dependant feature Y
Dataset_covid.survival[Dataset_covid.survival=='Y']=1
Dataset_covid.survival[Dataset_covid.survival=='N']=2
y_covid_0 = np.array(Dataset_covid.survival)
y_covid=np.zeros((134,1),dtype=int)
y_covid[:,0]=y_covid_0


Dataset_normal = pandas.read_csv('./Data_kaggle_normal.csv', low_memory=False)
Dataset_normal = Dataset_normal.iloc[:len(Dataset_covid)]

# Data_Kaggle_512_Normal
# creating independent features X and dependant feature Y
y_normal = np.zeros((len(Dataset_normal),1), dtype=int)


from skimage import io
from skimage.color import rgb2gray
from skimage.exposure import rescale_intensity

X_all=np.zeros((len(y_normal)+len(y_covid),512,512,1),dtype=np.uint8)
y_all=np.concatenate((y_covid, y_normal),axis=0)
X_all_names=[]

for i in range(len(y_covid)):
    FileName=Dataset_covid.values[i,5]
   # FileName=FileName+'.jpg'
    #print(FileName)    
    IMG = rgb2gray(io.imread('./Data_COVID_resized_selected/'+ FileName))
    
    if IMG.dtype=='float64':
#        print(IMG.dtype)
        IMG=rescale_intensity(IMG)*255
    else:
#        print(IMG.dtype)
        IMG=rescale_intensity(IMG)
              
    X_all[i,:,:,0]=IMG
    X_all_names.append(FileName)
    

for j in range(len(y_normal)):
    FileName=Dataset_normal.values[j,0]
    IMG = rgb2gray(io.imread('./Data_Kaggle_512_Normal/'+ FileName))
    if IMG.dtype=='float64':
#        print(IMG.dtype)
        IMG=rescale_intensity(IMG)*255
        #print(IMG.max(0))
    else:
#        print(IMG.dtype)
        IMG=rescale_intensity(IMG)
              
    X_all[len(y_covid)+j,:,:,0]=IMG
    X_all_names.append(FileName)
    
    # plt.imshow(X_all[130,:,:])
    


########################################
############## genreating input outputs from data #######
########################################

X_all_names=np.array(X_all_names)
TargetTensors=FnCreateTargetImages(y_all)
Y_all_tensors=TargetTensors
#plt.imshow(Y_all_tensors[1,:,:,:])
#print(y_all[1])
#plt.imshow(Y_all_tensors[2,:,:,:])
#print(y_all[2])


######################################################################################
################## NEtwork Architecture ##################################################
##########################################################################################

####################################### MRI FCN ###############################################
# mri FCN

In_shape=(X_all.shape[1],X_all.shape[2],1)

In = Input(shape=In_shape)

layer2D_encoder_1=Conv2D(filters=100, kernel_size=(3,3), strides=(2, 2), padding='valid', activation='linear', use_bias=True)(In)
layer2D_encoder_2=Conv2D(filters=50, kernel_size=(3,3), strides=(2, 2), padding='valid', activation='linear', use_bias=True)(layer2D_encoder_1)
layer2D_encoder_3=Conv2D(filters=50, kernel_size=(3,3), strides=(2, 2), padding='valid', activation='linear', use_bias=True)(layer2D_encoder_2)

layer_1D_layer=Flatten()(layer2D_encoder_3)

fc1 = Dense(50, kernel_initializer='normal', activation='linear')(layer_1D_layer)
#fc2 = Dense(50, kernel_initializer='normal', activation='linear')(fc1)

# interpretation layer
fc3 = Dense(100, activation='linear')(fc1)
hidden1 = Dropout(0.3)(fc3)
hidden1_reshape = Reshape((10, 10, 1))(hidden1)
e_2=tensorflow.keras.layers.BatchNormalization()(hidden1_reshape)
e_2=layer_lrelu(e_2)

layer2D_1 = Conv2DTranspose(filters=10, kernel_size=(3,3), strides=(1, 1),padding="same", activation='linear')(e_2)
layer2D_1=tensorflow.keras.layers.BatchNormalization()(layer2D_1)
layer2D_1=layer_lrelu(layer2D_1)

layer2D_2 = Conv2DTranspose(filters=10, kernel_size=(3,3), strides=(1, 1), dilation_rate=(2,2),padding="same", activation='linear')(e_2)
layer2D_2=tensorflow.keras.layers.BatchNormalization()(layer2D_2)
layer2D_2=layer_lrelu(layer2D_2)


layer2D_3 = Conv2DTranspose(filters=10, kernel_size=(3,3), strides=(1, 1), dilation_rate=(3,3), padding="same", activation='linear')(e_2)
layer2D_3=tensorflow.keras.layers.BatchNormalization()(layer2D_3)
layer2D_3=layer_lrelu(layer2D_3)

############################################################################
layer2D_4 = concatenate([layer2D_1,layer2D_2,layer2D_3])
layer2D_4=tensorflow.keras.layers.BatchNormalization()(layer2D_4)
############################################################################

layer2D_5 = Conv2DTranspose(filters=100, kernel_size=(3,3), strides=(2, 2), kernel_regularizer=tensorflow.keras.regularizers.l2(0.01), activation='linear')(layer2D_4)
layer2D_5=tensorflow.keras.layers.BatchNormalization()(layer2D_5)
layer2D_5=layer_lrelu(layer2D_5)

layer2D_5_2 = Conv2DTranspose(filters=100, kernel_size=(3,3), strides=(2, 2), kernel_regularizer=tensorflow.keras.regularizers.l2(0.01), activation='linear')(layer2D_5)
layer2D_5_2=tensorflow.keras.layers.BatchNormalization()(layer2D_5_2)
layer2D_5_2=layer_lrelu(layer2D_5_2)


layer2D_6 = Conv2DTranspose(filters=3, kernel_size=(3,3), strides=(1, 1), kernel_regularizer=tensorflow.keras.regularizers.l2(0.08), activation='linear' )(layer2D_5_2)
layer2D_6=tensorflow.keras.layers.BatchNormalization()(layer2D_6)

output_1 = tensorflow.keras.layers.Activation('relu')(layer2D_6)#
model_tensorization=Model(inputs= In, outputs=output_1)  

model_tensorization.summary()




#flat0=Flatten()(output_1)
## first feature extractor
#conv1 = Conv2D(i, kernel_size=3, activation='relu')(output_1)#relu
#conv1 = Dropout(0.4)(conv1)
#conv2 = Conv2D(j, kernel_size=3, activation='relu')(conv1)#relu
#conv2 = Dropout(0.4)(conv2)
#conv3 = Conv2D(k, kernel_size=3, activation='relu', name='last_features_2d')(conv2)#relu
#conv3 = Dropout(0.4)(conv3)
#flat1 = Flatten(name='last_features_flat')(conv3)
### cutting out from hidden1 output
## prediction output
##output_reg = Dense(4, activation='relu',kernel_regularizer=keras.regularizers.l1(0.01))(flat1)#relu
#outout_class=Dense(6, activation='softmax',kernel_regularizer=keras.regularizers.l1(0.01))(flat1)#softmax
#
##output_2=concatenate([output_reg, flat0, outout_class])
#model_classifier_1 = Model(inputs= [MRI_visible, PET_visible, COG_visible, CSF_visible, RF_visible], outputs=outout_class)  



##############################################################################
################## End NEtwork Architecture ##########################################
######################################################################################


##########################################################################################
################## Training on folding ###################################################
##########################################################################################


#OPTIMIZER_1=tensorflow.keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
OPTIMIZER_2=tensorflow.keras.optimizers.Adam(lr=0.001, beta_1=0.99, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
#OPTIMIZER_3=tensorflow.keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)


#model_classifier_1.save_weights('SavedInitialWeights.h5')
model_tensorization.save_weights('SavedInitialWeights_tensors.h5')

for repeator in range(0,1):

    kfold = StratifiedKFold(n_splits, shuffle=True, random_state=repeator)
    FoldCounter=0
    for train, test in kfold.split(X_all[:,0], y_all[:]):
        FoldCounter=FoldCounter+1   
        
#        model_classifier_1.load_weights('SavedInitialWeights.h5')        
        model_tensorization.load_weights('SavedInitialWeights_tensors.h5')        
        Y_train_here_4Net=Y_all_tensors[train,:,:,:]
        X_train_here_4Net=X_all[train,:]
        X_train_here_names=X_all_names[train]
        #print(X_train_here_4Net.shape)
        
        
        print('---Repeat No:  ', repeator+1, '  ---Fold No:  ', FoldCounter)        
        
        
        model_tensorization.compile(loss=custom_loss_2, optimizer=OPTIMIZER_2)
        History = model_tensorization.fit(X_train_here_4Net, Y_train_here_4Net, validation_split=0.1,  epochs=a.max_epochs, batch_size=a.BatchSize, verbose=1)#250-250
        # model_tensorization.compile(loss=custom_loss_1, optimizer=OPTIMIZER_1)
        # History = model_tensorization.fit(X_train_here_4Net, Y_train_here_4Net, validation_split=0.01,  epochs= 250, batch_size=BatchSize, verbose=1)#250-250

        # summarize history for loss
        plt.plot(History.history['loss'])
        plt.plot(History.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.grid()
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig(a.output+'Fold_'+str(FoldCounter)+'_History.png')
        plt.close()


        for i in range(50):#range(len(train)):
            plt.figure()
            plt.subplot(122)
            plt.imshow(((model_tensorization.predict(X_train_here_4Net)[i,:,:,:])))
            plt.subplot(121)
            plt.imshow(Y_train_here_4Net[i,:,:,:])
            plt.savefig(a.output+'Fold_'+str(FoldCounter)+'_Trained_'+X_train_here_names[i], dpi=300)
            plt.close()


        X_test_here_4Net=X_all[test,:]
        Y_test_here_4Net=Y_all_tensors[test,:,:,:]
        X_test_here_names=X_all_names[test]
        
        for i in range(len(test)):#range(len(test)):            
            fig = plt.figure()
            ax = fig.gca()
            
            plt.subplot(122)
            plt.imshow(((model_tensorization.predict(X_test_here_4Net)[i,:,:,:])))

            plt.subplot(121)
            plt.imshow(Y_test_here_4Net[i,:,:,:])
            
            plt.savefig(a.output+'Fold_'+str(FoldCounter)+ '_Test_'+X_test_here_names[i], dpi=300)
            plt.close()

            # Hide axes ticks
            ax.set_xticks([])
            ax.set_yticks([])            

       
