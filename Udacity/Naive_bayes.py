#!/usr/bin/python

""" 
    this is the code to accompany the Lesson 1 (Naive Bayes) mini-project 

    use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1

"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.metrics import accuracy_score


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




    ### import the sklearn module for GaussianNB
from sklearn.naive_bayes import GaussianNB
### create classifier
t0 = time()
clf = GaussianNB()
print "training time:", round(time()-t0, 3), "s"

    ### fit the classifier on the training features and labels
clf.fit(features_train, labels_train)

    ### use the trained classifier to predict labels for the test features

t1= time()
pred = clf.predict(features_test)
print "testing time:", round(time()-t1, 3), "s"

    ### calculate and return the accuracy on the test data


accuracy = accuracy_score(pred,labels_test)
print "accuracy=", accuracy



#########################################################


