import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Datasets/Question 3")

traindata = np.genfromtxt(os.path.join(dataset_path,"mnist_train.csv"), delimiter=',')
testdata = np.genfromtxt(os.path.join(dataset_path,"mnist_test.csv"), delimiter=',')

lambda_pows=[-10, -8, -6, -4, -2, 0]
classes = [i for i in range(10)]

def compute_cost(W, X, Y, l):
    # calculate hinge loss
    N = X.shape[0]
    distances = 1 - Y * (np.dot(X, W))
    distances[distances < 0] = 0 
    hinge_loss = (np.sum(distances) / N)
    cost = l * np.dot(W, W) + hinge_loss
    return cost

def sgd(features, outputs):
    max_epochs = 5000
    weights = np.random.normal(loc=0, scale=0.01, size=features.shape[1])
    nth = 0
    prev_cost = float("inf")
    cost_threshold = 0.01  
    
    for epoch in range(1, max_epochs):
        
        X, Y = shuffle(features, outputs)
        for ind, x in enumerate(X):
            ascent = calculate_cost_gradient(weights, x, Y[ind])
            weights = weights - (learning_rate * ascent)

        
        if epoch == 2 ** nth or epoch == max_epochs - 1:
            cost = compute_cost(weights, features, outputs)
            print("Epoch is: {} and Cost is: {}".format(epoch, cost))
            # stoppage criterion
            if abs(prev_cost - cost) < cost_threshold * prev_cost:
                return weights
            prev_cost = cost
            nth += 1
    return weights
            

            
