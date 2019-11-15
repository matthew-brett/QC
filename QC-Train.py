# import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions
from random import shuffle
from keras import layers
from sklearn.cross_validation import KFold,StratifiedKFold
import  sklearn.cross_validation
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from keras.models import load_model
import os
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--axial', action='store_true')
    parser.add_argument('--sagittal', action='store_true')
    args = parser.parse_args()
    return args




train_dir = './Data/Axial/'
Res_dir='./Results/Axial/'

#VGG-Axial
l1=3
l2=3
l3=512
##################
#VGG-Axial
a1=116
a2=116
args = get_args()   

if (args.sagittal==1):
     ##VGG-Sagittal
     l1=2
     l2=3
     l3=512
     ##VGG-Sagittal
     a1=75
     a2=110
     train_dir = './Data/Sagittal/'
     Res_dir='./Results/Sagittal/'

def creport (yt, y_pred):
    cm1 = confusion_matrix(yt, y_pred)

    total1 = sum(sum(cm1))
    accuracy1 = float(cm1[0, 0] + cm1[1, 1]) / total1
    recall1 = float(cm1[0, 0]) / (cm1[0, 0] + cm1[0, 1])
    precision1 = float(cm1[0, 0]) / (cm1[0, 0] + cm1[1, 0])
    return accuracy1,recall1,precision1



nTrain = sum([len(files) for r, d, files in os.walk(train_dir)])
nVal=nTrain/5



datagen = keras.preprocessing.image.ImageDataGenerator(
      rescale=1.0/255,
      featurewise_center=True,
      featurewise_std_normalization=True,
      width_shift_range=0.2,
      height_shift_range=0.2,
      rotation_range=20, horizontal_flip=True,brightness_range=[0.8,1.2], shear_range=0.2,
      zoom_range=[0.8,1.2]
)

batch_size = 20


TrasN='New-Model'



train_features = np.zeros(shape=(nTrain, l1, l2, l3))
train_labels = np.zeros(shape=(nTrain, 2))

 
vgg_conv =  keras.applications .VGG16(weights='imagenet',
                  include_top=False,
                 input_shape=(a1, a2, 3))
#

train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(a1, a2),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True)

i = 0
file_name=train_generator.filenames


model = keras.models.Sequential()
model.add(keras.layers.Dense(256, activation='relu', input_dim=l1* l2 * l3))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(2, activation='softmax'))

model.compile(optimizer=keras.optimizers.RMSprop(lr=2e-4),
              loss='binary_crossentropy',
              metrics=['acc'])
kf = sklearn.cross_validation.KFold(np.shape(train_features)[0], n_folds=5, shuffle=True, random_state=100)

for inputs_batch, labels_batch in train_generator:
  
    features_batch = vgg_conv.predict(inputs_batch)
    print (np.shape(features_batch))
    train_features[i * batch_size: (i + 1) * batch_size] = features_batch
    train_labels[i * batch_size: (i + 1) * batch_size] = labels_batch
    i += 1
    print (i * batch_size)
    if i * batch_size >= nTrain:
        break
    print (np.shape(file_name))
train_features = np.reshape(train_features, (nTrain, l1 * l2 * l3))

Acc=[]
Rec=[]
Prec=[]
F1=[]


outfile_acc=Res_dir+TrasN+'ValDetailedRes-'+'.txt'
ff=open(outfile_acc,"w")
i=0;
for train, test in kf:
    X_train, X_test, y_train, y_test =train_features[train], train_features[test], train_labels[train], train_labels[test]
    history = model.fit(X_train,
                   y_train,
                   epochs=20,
                   batch_size=batch_size,
                  validation_data=(X_test,y_test))


    predictions = model.predict_classes(X_test)

    
    
    model.save(Res_dir+TrasN+'-'+str(i)+'.h5')
    i=i+1
    accuracy1, recall1, precision1= creport(y_test[:,1], predictions)
    Acc.append(accuracy1)
    Rec.append(recall1)
    Prec.append(precision1)


  
    
    

     
            
tosave_acc =' Alg ' +TrasN \
+ '\nMeanacc: ' + str(np.mean(Acc)) + ', STDAcc: ' + str(np.std(Acc)) \
+ '\nMeanRecall: ' + str(np.mean(Rec)) + ', STDRecall: ' + str(np.std(Rec)) \
+ '\nMeanPrecision: ' + str(np.mean(Prec)) + ', STDPrecision: ' + str(np.std(Prec)) \
+ '\n'

ff.write(tosave_acc)





ff.flush()
ff.close()