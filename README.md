### Sparse Multivariate Factor Regression
#### A python implementation of [Sparse Multivariate Factor Regression](http://arxiv.org/abs/1502.07334) - Milad Kharratzadeh, Mark Coates

##### Import needed libraries.
```python
import numpy as np
import smfr
```
##### For initial test of smfr we will just use two random matrices.
```python
num_data_points = 50
num_features = 12
num_stations = 20
m = 10
X = np.random.rand(num_data_points, num_features)
A = np.random.rand(num_features, m)
B = np.random.rand(m, num_stations)
```
##### We are using this y so we will know the correct answer. In normal use y will be the target data.
```python
y = np.dot(X, np.dot(A,B))
lambda_1 = 0.1
lambda_2 = 0.1
epsilon = 0.001  
```
##### Initialize the SMFR problem.
```python
model = smfr.SMFR(X, y, lambda_1, lambda_2, m, epsilon)
```
##### Fit the model.
```python
model.fit()
```
##### Access our solutions.
```python
print model.A
print model.B
print model.m
```
##### We can also make predictions with our model.
```python
X_test = np.random.rand(num_data_points, num_features)
y_test = model.predict(X_test)
```