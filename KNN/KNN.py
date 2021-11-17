import pandas as pd
import numpy as np
import math
import os

dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Datasets/Question 4")

traindata = np.genfromtxt(os.path.join(dataset_path,"train_data.csv"), delimiter=',')
testdata = np.genfromtxt(os.path.join(dataset_path,"test_data.csv"), delimiter=',')
testlabels = np.genfromtxt(os.path.join(dataset_path,"test_labels.csv"), delimiter=',')
trainlabels = np.genfromtxt(os.path.join(dataset_path,"train_labels.csv"), delimiter=',')
klist = [2, 5, 10, 50]

test_index = len(testdata)
train_index = len(traindata)
trdmax = np.amax(traindata, axis=0)
trdmin = np.min(traindata, axis=0)
trdmean = np.mean(traindata, axis=0)
newtraindata = (traindata - trdmean)/(trdmax - trdmin)
newtestdata = (testdata - trdmean)/(trdmax - trdmin)

def d_one(a, b):
    c = abs(a - b)
    d = np.max(c, axis=1)
    return d

def d_two(a, b):
    c = abs(a - b)
    d = np.sum(c, axis=1)
    return d

for k in klist:
    truenum = 0
    for i in range(test_index):
        
        b = d_one(newtraindata, newtestdata[i])
        c = np.argsort(b)
        neighbors = [trainlabels[c[l]] for l in range(k)]
        d = max(set(neighbors), key=neighbors.count)
        if(d == testlabels[i]):
            truenum += 1
    
    trueperc = truenum/test_index * 100
    print("The accuracy for k = {} = {} percent".format(k, trueperc))



