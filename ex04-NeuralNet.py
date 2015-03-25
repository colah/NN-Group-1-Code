from Network import Network, InputLayer, FullyConnectedLayer, SoftMaxLayer, sigmoid, log_likelihood
import data

import math
import numpy


mnist = data.load("mnist", flatvecs = True)
test_data = mnist["test"].values()
train_data = mnist["train"]

net = Network(
    layers = [InputLayer(784),
              FullyConnectedLayer(20, activation = sigmoid),
              SoftMaxLayer(10)],
    cost=log_likelihood)

for i in range(100):
    for m in range(100):
        net.train(*train_data.get_batch(30), learning_rate = 0.01 )
    print i, "Test Accuracy:", net.accuracy(*test_data)

