import numpy as np
import smfr
from sklearn import datasets
import pandas as pd
from ggplot import *

# The digits dataset
# digits = datasets.load_digits()
# n_samples = len(digits.images)
# data = digits.images.reshape((n_samples, -1))
# X = data[:n_samples / 2]
# y = np.zeros((n_samples/2, 1))
# y[:,0] = digits.target[:n_samples / 2]

np.random.seed(0)
# For initial test of smfr we will just use two random matrices
num_data_points = 50
num_features = 12
num_stations = 20
m = 5

X = np.random.rand(num_data_points, num_features)
# y = np.random.rand(num_data_points, num_stations)

A = np.random.rand(num_features, m)
B = np.random.rand(m, num_stations)

# now
y = np.dot(X, np.dot(A,B))

lambda_1 = 0.1
lambda_2 = 0.1
epsilon = 0.001

print (0.5)*np.power(np.linalg.norm(y - np.dot(X, np.dot(A, B)), ord='fro'), 2) + lambda_1*np.sum(np.abs(A)) + lambda_2*np.sum(np.abs(B))
  

# Initialize the SMFR problem
model = smfr.SMFR(X, y, lambda_1, lambda_2, m, epsilon)

# Fit the model
model.fit()

print "model.A:"
print model.A
print "A:"
print A
print "model.B:"
print model.B
print "B:"
print B

print "AB"
print np.dot(A,B)
print"modelA dot modelB"
print np.dot(model.A, model.B)

f_ABs = pd.DataFrame(model.f_ABs)
delta_f_ABs = pd.DataFrame(model.delta_f_ABs)
print f_ABs
print delta_f_ABs

p = ggplot(aes(x='Iteration', y='0'), data=f_ABs)
print p + geom_point() + geom_line() + stat_smooth(color='blue')
