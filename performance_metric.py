# -*- coding: utf-8 -*-
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
from sklearn.datasets import load_files
from imageio import imread, imsave
import os

# %%

# import some data to play with
# iris = datasets.load_iris()
# X = iris.data
# y = iris.target

curDir = "./fruits-360/Training"

trueFiles = load_files(curDir)

data2 = trueFiles.data


X = []
y = trueFiles.target
for i in range(len(data2)):
    X.append(imread(data2[i]).flatten())
    
class_names = [os.path.basename(x[0]) for x in os.walk(curDir)]
class_names = class_names[1:len(class_names)]

# %%

# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Run classifier, using a model that is too regularized (C too low) to see
# the impact on the results
classifier = svm.SVC(kernel='linear', C=0.01).fit(X_train, y_train)

np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(classifier, X_test, y_test,
                                 display_labels=trueFiles.target_names,
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)

plt.show()
