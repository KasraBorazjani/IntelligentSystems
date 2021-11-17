import numpy as np
import network
import math
import os


dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "DataSet_Q2/DataSet2")
train_data = np.genfromtxt(os.path.join(dataset_path,"Reduced_Train_Data.csv"), delimiter=',', dtype="|U5")
train_label_raw = np.genfromtxt(os.path.join(dataset_path,"trainLabels.csv"), delimiter=',', dtype="|U5")
test_data = np.genfromtxt(os.path.join(dataset_path,"Reduced_Test_Data.csv"), delimiter=',', dtype="|U5")
test_label_raw = np.genfromtxt(os.path.join(dataset_path,"testLabels.csv"), delimiter=',', dtype="|U5")

mynewnn = network.network()
mynewnn.add(network.fcLayer(128, 150))
mynewnn.add(network.activationLayer(network.tanh, network.tanh_prime))
mynewnn.add(network.fcLayer(150,10))
mynewnn.add(network.activationLayer(network.tanh, network.tanh_prime))

mynewnn.use(network.mse, network.mse_prime)


train_data = train_data.astype(float)
train_label_raw = train_label_raw.astype(int)
train_label = np.eye(10)[train_label_raw]
train_label = train_label.astype(float)

# test_train_data = train_data[0:1]
# print("test data is: {} \n".format(test_train_data))
# test_train_label = train_label[0:1]
# print("test label is: {} \n".format(test_train_label))

mynewnn.fit(train_data, train_label, epochs=5, learning_rate=0.9)
# test_data = test_data.astype(float)
# test_label_raw = test_label_raw.astype(int)
# test_label = np.eye(10)[test_label_raw]
# test_label = test_label.astype(float)
# test_results = mynewnn.predict(test_data)


# print("test results[0] = {}".format(test_results[0]))

# np.savetxt("test_results.csv", test_results, delimiter=",")
# counter = 0
# for i in range(len(test_label)):
#     if (test_results[i]==test_label[i]):
#         counter += 1
# accuracy = counter/len(test_label) * 100

# print("test accuracy = {}".format(accuracy))