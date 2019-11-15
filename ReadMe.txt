# Zahra Riahi Samani
# Center for Biomedical Image Computing and Analytics: CBICA
# University of Pennsylvania
# November 2019

Zahra RiahiSamani*, Jacob A. Alappatt, Drew Parker, Abdol Aziz Ould Ismail, Ragini Verma,
QC-Automator: Deep Learning-based Automated Quality Control for Diffusion MR Image
arxiv:
 

This directory includes:
QCTest.py:  It loads the pre-trained models and run in on the sample of test data.
QCTrain.py: It can be used to train one own model.
Models:     It contains pre-trained models.
Results:    It is the place where the result is written.
Test:       It contains the test data. 
Data        Training data should be placed here in order to use the QCTrain script.The data set should contain 2-d slices of artifact-free and artifactual slices which are converted to jpg format. 




Requirenment:
you need to install these packages:
keras, tensorflow, dipy, cv2, PIL, Sklearn

How to run:
QCTest --axial will do pre-processing and run the model on axial data, prints the accuracy,precision and recall and saves the output lables at the Results folder. The default is axial.
QCTest --saggital will do pre-processing and run the model on sagittal data, prints the accuracy, precision and recall and saves the output lables at Results folder.

Accuracy, precision and recall are calculated by averaging among 5 different models trained during a 5-fold keras validation process.
Output labels contains three columns: imagename, true lable, predicted labels which is the average between the 5 models trained during the 5 fold cross validation.
Output labels are interpretd in this way:
0: no artifact
1: artifact present


QCTrain --axial    will train the axial models based on the trining data  which should be stored in Data folder. The default is axial.
QCTrain  --sagittal will train the sagittal models based on the trining data  which should be stored in Data folder


The expected output for QCTest --axial:
Found 98 images belonging to 2 classes.
Computing features
features computed
Running Axial Model:  0
Running Axial Model:  1
Running Axial Model:  2
Running Axial Model:  3
Running Axial Model:  4
Accurcay: 0.978
Recall: 0.963
Precision: 0.992

The expected output for QCTest --saggital:

Found 98 images belonging to 2 classes.
Computing features
features computed
Running Sagittal Model:  0
Running Sagittal Model:  1
Running Sagittal Model:  2
Running Sagittal Model:  3
Running Sagittal Model:  4
Accurcay: 0.939
Recall: 0.971
Precision: 0.913