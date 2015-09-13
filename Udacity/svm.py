#!/usr/bin/python

""" 
    this is the code to accompany the Lesson 2 (SVM) mini-project

    use an SVM to identify emails from the Enron corpus by their authors
    
    Sara has label 0
    Chris has label 1

"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

features_train, features_test, labels_train, labels_test = preprocess()

#########################################################

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels

### import the sklearn module for GaussianNB
from sklearn.naive_bayes import GaussianNB

### create classifier

clf = SVC(kernel="rbf",C=10000)


    ### fit the classifier on the training features and labels
t0 = time()
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"
    ### use the trained classifier to predict labels for the test features

t1= time()
pred = clf.predict(features_test)
print "testing time:", round(time()-t1, 3), "s"
count = 0
for result in pred:
    if result == 1:
      count =count + 1

    ### calculate and return the accuracy on the test data

accuracy = accuracy_score(pred,labels_test)
print "accuracy=", accuracy
print count



#########################################################




