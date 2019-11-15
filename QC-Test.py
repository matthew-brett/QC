# import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras
from keras.applications.imagenet_utils import decode_predictions
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from keras.models import load_model
import os
import argparse
import csv

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--axial', action='store_true')
    parser.add_argument('--sagittal', action='store_true')
    args = parser.parse_args()
    return args


batch_size=20
val_dir='./Test-Data/Axial/'
model_dir='./Models/Axial/'
Res_dir='./Results/Axial/'
model_type='Axial'

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
     model_dir='./Models/Sagittal/'
     Res_dir='./Results/Sagittal/'
     val_dir='./Test-Data/Sagittal/'
     model_type='Sagittal'


def creport (yt, y_pred):
    cm1 = confusion_matrix(yt, y_pred)
    total1 = sum(sum(cm1))
    accuracy1 = float(cm1[0, 0] + cm1[1, 1]) / total1
    recall1 = float(cm1[0, 0]) / (cm1[0, 0] + cm1[0, 1])
    precision1 = float(cm1[0, 0]) / (cm1[0, 0] + cm1[1, 0])
    return accuracy1,recall1,precision1

 
nval = sum([len(files) for r, d, files in os.walk(val_dir)])
TrasN='Model'


datagen = keras.preprocessing.image.ImageDataGenerator(        rescale=1.0/255,featurewise_center=True,
                                                       featurewise_std_normalization=True)

vgg_conv =  keras.applications .VGG16(weights='imagenet', include_top=False, input_shape=(a1, a2, 3))



val_features = np.zeros(shape=(nval, l1, l2, l3))
val_labels = np.zeros(shape=(nval, 2))

               
val_generator = datagen.flow_from_directory(
    val_dir,
    target_size=(a1, a2),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False)

file_name=val_generator.filenames
i = 0
print ('Computing features')
  
for inputs_batch, labels_batch in val_generator:
   features_batch = vgg_conv.predict(inputs_batch)
  
   val_features[i * batch_size: (i + 1) * batch_size] = features_batch
   val_labels[i * batch_size: (i + 1) * batch_size] = labels_batch
   i += 1
   if i * batch_size >= nval:
       break
  

val_features = np.reshape(val_features, (nval, l1 * l2 * l3))
print ('features computed')

Acc=[]
Rec=[]
Prec=[]

CF = open(Res_dir+TrasN +"TestSet-PredictedVsTrue.csv", "w")
Cwriter = csv.writer(CF, delimiter=',')
Y=np.zeros([nval])
nfolds=5
for i in range(nfolds):
    
    model=load_model(model_dir+TrasN+'-'+str(i)+'.h5')
    print ('Running '+model_type+ ' Model:  '+str(i))
    predictions = model.predict_classes(val_features) 
    Y=Y+predictions
    
    yt=val_labels[:,1]
    accuracy1, recall1, precision1= creport(yt, predictions)
    Rec.append(recall1)
    Prec.append(precision1)
    Acc.append(accuracy1)
    

Y2=Y/nfolds
#Binarization
Y[np.where(Y2<=0.5)]=1
Y[np.where(Y2>0.5)]=0
##reversing labels
yt=np.ones(np.shape(yt))-yt

for i in range(len(predictions)):
    Cwriter.writerow([file_name[i],int(yt[i]),int(Y[i])])
   

tosave_acc =  'Accurcay: ' + str(round(np.mean(Acc),3)) + '\n'+ 'Recall: ' + str(round(np.mean(Rec),3)) + '\n' + 'Precision: ' + str(round(np.mean(Prec),3)) + '\n' 
print(tosave_acc)

CF.flush()
CF.close()