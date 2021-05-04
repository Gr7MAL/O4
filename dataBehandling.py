# -*- coding: utf-8 -*-
"""
Created on Tue May  4 12:52:47 2021

@author: bjark
"""

from sklearn.datasets import load_files
from os import getcwd
from imageio import imread, imsave
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

#%% Load data
curDir = getcwd()

dataSet = load_files(curDir, shuffle=False)

#%%

data = dataSet.data

testImg = data[0]

img = imread(testImg)

#%%

imageArray = []

for i in range(len(data)):
    imageArray.append(imread(data[i]))
    
#%%

flatImages = []

for i in range(len(data)):
    flatImages.append(imread(data[i]).flatten())

#%% Try with 2 components to visualize
pca = PCA(n_components=2)

pca.fit(flatImages)

inputData = pca.transform(flatImages)

#%%

v1 = pca.components_
plt.plot(pca.explained_variance_)
plt.title("Varianse")
plt.show()

#%% Projektion på 2D til visualisering

plt.scatter(inputData[:,0], inputData[:,1])
plt.title("Data plot with 2 components")

#%% Plot based on type and color
n = 492;
applesGS = inputData[0:n-1,:]
applesRED = inputData[n:(2*n)-1,:]
banana = inputData[2*n:(3*n)-1,:]
eggplant = inputData[3*n:(4*n)-1,:]
lemon = inputData[4*n:(5*n)-1,:]
mango = inputData[5*n:(6*n)-1,:]

plt.scatter(applesGS[:,0], applesGS[:,1],color='g',linewidths=0)
plt.scatter(applesRED[:,0], applesRED[:,1],color='r',linewidths=0)
plt.scatter(banana[:,0], banana[:,1],color='y',linewidths=0)
plt.scatter(eggplant[:,0], eggplant[:,1],color='m',linewidths=0)
plt.scatter(lemon[:,0], lemon[:,1],color='b',linewidths=0)
plt.scatter(mango[:,0], mango[:,1],color='c',linewidths=0)
plt.legend(dataSet.target_names)
plt.title("Data sorted")
#This show that we need more components to classify the data.

#%% Test with real data
curDir = getcwd()
trueFiles = load_files(curDir)

data2 = trueFiles.data


X = []
y = trueFiles.target
for i in range(len(data2)):
    X.append(imread(data2[i]).flatten())




#%% Now with 20 components to visualize
pca2 = PCA(n_components=8)

pca2.fit(X)

processedX = pca2.transform(X)

#%% Split in trainning and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train ,y_test = train_test_split(processedX,y,test_size=0.2)

#%%
import sklearn.svm as svm
clf = svm.SVC()

clf.fit(X_train,y_train)

#%%

y_pred = clf.predict(X_test)

s = clf.score(X_test,y_test)