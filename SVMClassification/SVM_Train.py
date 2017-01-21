# -*- coding: utf-8 -*-
"""
SVM Train
@author Gaurav
"""

import cv2;
import os;
from sklearn.externals import joblib;
import numpy as np;
from sklearn import svm;



# Extracting Histograms and assigning label to each class
histograms = [];
labels = [];
lNo = 0;

for folder in os.listdir(r'U:\Study\VisualDatabases\Project\TrainingImages'):
    lNo = lNo +1; 
    fNo = 0;
    for files in os.listdir("U:\Study\VisualDatabases\Project\TrainingImages/"+folder):
        print (str(lNo)+" "+folder+" "+files);
        fNo = fNo +1;
        if(fNo > 1000):
            break;
        image = cv2.imread("U:\Study\VisualDatabases\Project\TrainingImages/"+folder+"/"+files ,0);
        hist = cv2.calcHist([image],[0],None,[16],[0,256]);
        histograms.append([hist[0][0],hist[1][0],hist[2][0],hist[3][0],hist[4][0],hist[5][0],hist[6][0],hist[7][0],hist[8][0],hist[9][0],hist[10][0],hist[11][0],hist[12][0],hist[13][0],hist[14][0],hist[15][0]]);
        labels.append(lNo);
 
histograms = np.asarray(histograms);   


#Train the data for creating a model
print "Training Started";

clf = svm.LinearSVC();
clf.fit(histograms,labels);
print "Training Finished";

#Dump the model into a file
joblib.dump(clf, r'U:\Study\VisualDatabases\Project\TrainedModel\ModelSVC.pkl')

 
