# Import needed variables
import cvxpy as cvx
import numpy as np

# Ensure we can repeat random
np.random.seed(0)

# Generate our neede matrices
num_data_points = 1000
num_stations = 150
num_features = 30
m_rank = 15



XA = np.random.rand(num_data_points, 10).dot(np.random.rand(10, m_rank))
y = np.random.rand(num_data_points, num_stations)
print XA.shape
print y.shape

# Generate our needed lambda value
lambda_1 = 0.001

# Declare our initial variable
Bv = cvx.Variable(m_rank, num_stations)

# Declare our objective function, and constraints
obj = cvx.Minimize((1/2)*cvx.norm(y - XA*Bv, 'fro') + lambda_1*cvx.norm(Bv, 1))
constraints = []

# Solve the convex problem
prob = cvx.Problem(obj, constraints)
prob.solve(solver=cvx.SCS)

# Update B value
B = Bv.value
print B