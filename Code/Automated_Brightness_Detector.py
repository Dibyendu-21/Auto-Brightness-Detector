# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 11:43:11 2019

@author: Sonu
"""

import cv2 as cv
import numpy as np
import os
import pandas as pd
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from sklearn.cross_validation import train_test_split
from keras.models import Sequential
import keras
from keras.models import load_model
from custom_metric import fmeasure
#Finding Brightness of a sample Image
#img = cv.imread('train-00005.jpg')
#bright=np.mean(img)
#print(bright)

#Function to read a set of images from the dataset, calculate the true brightness of each of the images, returns image,filename and brightness
def read_image_calculate_brightness(file):
    df=pd.read_csv(file)
    a=df.loc[:,'fname']
    filename = np.empty(len(a), dtype=object)
    true_brightness = np.empty(len(a), dtype=object)
    k=0

    for i, v in a.items():
        filename[k]=df.loc[i ,'fname']
        k=k+1


    imgpath = np.empty(len(a), dtype=object)
    images =  np.empty(len(a), dtype=object)

    for n in range(0, len(filename)):
            #Specify path where test images are stored. 
            imgpath[n] = os.path.join(r"C:\Users\Sonu\Documents\M.TECH\Research\UCLA-protest\img\new\demo_train",filename[n])
            images[n] = cv.imread(imgpath[n])
            images[n] = cv.resize(images[n], (150, 150))
            true_brightness[n] = np.mean(images[n])
    return images,true_brightness,filename         

Image,Continous_Brightness,Filename=read_image_calculate_brightness(r'C:\Users\Sonu\Documents\M.TECH\Research\UCLA-protest\img\new\demo_train\train_tf.csv')

#Creating a new csv file with the filename and their corresponding brightness value
File_Name = pd.DataFrame(Filename, columns=['File_Name'])  
Bright = pd.DataFrame(Continous_Brightness, columns=['Contionus_Brightness'])
Total = pd.concat([File_Name, Bright], axis=1)
if not os.path.isdir('Code_Challenge'):
    os.mkdir('Code_Challenge')
sub_file = os.path.join('Code_Challenge', 'Calculate_Brightness' + '.csv')
Total.to_csv(sub_file, index=False)


      
#Binning the true brightness value in the range of 0-10 and creating a new csv file   
df=pd.read_csv(r'C:\Users\Sonu\Documents\M.TECH\Seminar\Code_Challenge\Calculate_Brightness.csv')  
df['Discrete_Brightness'] = pd.cut(df['Contionus_Brightness'],[0,23.18,46.36,69.54,92.72,115.9,139.08,162.26,185.44,208.62,231.8,255], labels=[0,1,2,3,4,5,6,7,8,9,10])
sub_file = os.path.join('Code_Challenge', 'Discrete_Brightness' + '.csv')
df.to_csv(sub_file, index=False)

#Final Dataset used for training. Need not be executed at the test site.
def read_image_label_for_training(file):
    df=pd.read_csv(file)
    a=df.loc[:,'File_Name']
    filename = np.empty(len(a), dtype=object)
    k=0

    for i, v in a.items():
        filename[k]=df.loc[i ,'File_Name']
        k=k+1

    label = df.loc[:,'Discrete_Brightness']
    imgpath = np.empty(len(a), dtype=object)
    images =  np.empty(len(a), dtype=object)

    for n in range(0, len(filename)):
            imgpath[n] = os.path.join(r"C:\Users\Sonu\Documents\M.TECH\Research\UCLA-protest\img\new\demo_train",filename[n])
            images[n] = cv.imread(imgpath[n])
            images[n] = cv.resize(images[n], (150, 150))
            
    return images,label         

Image,label=read_image_label_for_training(r'C:\Users\Sonu\Documents\M.TECH\Seminar\Code_Challenge\Discrete_Brightness.csv')

img_rows=150
img_cols=150
data=[]

for img in Image:
    data.append(img)

print('label is',label)

#Converting labels to one-hot encoded form    
def indices_to_one_hot(label, nb_classes):
    targets = np.array(label).reshape(-1)
    return np.eye(nb_classes)[targets]


one_hot_label = indices_to_one_hot(label, 11)


'''
#Function to split data into train,test and validation set    
def split_validation_set_with_hold_out(train, target, test_size):
    random_state = 51
    train, X_test, target, y_test = train_test_split(train, target, test_size=test_size, random_state=random_state)
    X_train, X_holdout, y_train, y_holdout = train_test_split(train, target, test_size=test_size, random_state=random_state)
    return X_train, X_test, X_holdout, y_train, y_test, y_holdout    

 
data_train, data_test, data_holdout, label_train, label_test, label_holdout = split_validation_set_with_hold_out(data,one_hot_label,0.2)
    


if K.image_data_format() == 'channels_first':
    input_shape = (3, img_rows, img_cols)
else:
    input_shape = (img_rows, img_cols, 3)
    
#Model_building
model = Sequential()
model.add(Conv2D(4, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(8, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(16, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dropout(0.4))
model.add(Dense(11))
model.add(Activation('softmax'))

#Model Configuration
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=[fmeasure])
fmeasure_check=keras.callbacks.ModelCheckpoint(filepath='.\Brightness_Model.h5',monitor="val_fmeasure", verbose=0, save_best_only=True, mode='max')

#Model Training
model.fit(np.array(data_train),np.array(label_train), batch_size=10, nb_epoch=50, verbose=1, validation_data=(np.array(data_holdout),np.array(label_holdout)),callbacks=[fmeasure_check])
score = model.evaluate(np.array(data_test),np.array(label_test), verbose=0)
print(score)

#Loading the model and testing it
file_path='.\Brightness_Model.h5'

#def testing(file_path,data_test,label_test):
#    model1 = load_model(file_path,custom_objects={'fmeasure': fmeasure})
#    new_score=model1.evaluate(np.array(data_test),np.array(label_test), verbose=0)
#    return new_score

#score = testing(file_path,data_test,label_test)
#print(score)
'''    