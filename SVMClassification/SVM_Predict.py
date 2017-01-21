# -*- coding: utf-8 -*-
"""
Created on Mon Dec 05 20:42:25 2016

@author: Gdams
"""

import cv2;
import os;
from sklearn.externals import joblib;
import numpy as np;


names = [];
histograms = [];
for files in os.listdir(r'U:\Study\VisualDatabases\Project\Faces'):
        image = cv2.imread("U:\Study\VisualDatabases\Project\Faces/"+files ,0);
        hist = cv2.calcHist([image],[0],None,[4],[0,256]);
        histograms.append([hist[0][0],hist[1][0],hist[2][0],hist[3][0],hist[4][0],hist[5][0],hist[6][0],hist[7][0],hist[8][0],hist[9][0],hist[10][0],hist[11][0],hist[12][0],hist[13][0],hist[14][0],hist[15][0]]);
        names.append(files);
        
histograms = np.asarray(histograms);

clf = joblib.load(r'U:\Study\VisualDatabases\Project\TrainedModel\ModelSVC.pkl');

prediction = clf.predict(histograms);

for sample,label in zip(names,prediction):
    print(sample+" | "+str(label));


        