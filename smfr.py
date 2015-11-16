# Michael Curtis 260475694
# Sparse Multivariate Factor Regression
import numpy as np
from sklearn import linear_model
import cvxpy as cvx
import pandas as pd

# First we will define a class for smfr to initialize needed variables for regression. Our initial implementation will ignore lambdas (set lambda 1 and 2 to 0).
class SMFR:
	
	# Initialize inputs to local variables, eventually this will include an lambda 1 and 2. Input assumes numpy matrices
	def __init__(self, X, y, lambda_1, lambda_2, m, epsilon):
        
		self.X = X
		self.y = y
		self.lambda_1 = lambda_1
		self.lambda_2 = lambda_2
		self.epsilon = epsilon
		self.m = m
		
		# Define the lasso model we will need for alternating minimization to find B
		# self.clf_B = linear_model.Lasso(alpha=self.lambda_2)

		# X: n x p = num_data_points x num_features
		# y: n x q = num_data_points x num_stations
		# A: p x m = num_features x rank
		# B: m x q = rank x num_stations
		self.num_data_points = X.shape[0]
		self.num_features = X.shape[1]
		self.num_stations = y.shape[1]

		# Need to initialize our weight matrices A and B, A0, B0 will be random N(0,1). Might have to convert these into np.matrix
		self.A = np.random.rand(self.num_features, self.m)
		self.B = np.random.rand(self.m, self.num_stations)

		# In this option we will create a randomly generated array of 0's or 1's
		# assign_zero_ones = np.vectorize(lambda x : 1 if x > 0.5 else 0)
		# self.A = assign_zero_ones(self.A)
		# self.B = assign_zero_ones(self.B)
		
		# Now assign y such that y = XAB
		self.y = np.dot(self.X, np.dot(self.A, self.B))
		
		# Initiallizing the evaluation value, this will always be the previous value		
		self.f_of_A_B = (0.5)*np.power(np.linalg.norm(self.y - np.dot(self.X, np.dot(self.A, self.B)), ord='fro'), 2) + self.lambda_1*np.sum(np.abs(self.A)) + self.lambda_2*np.sum(np.abs(self.B))
  
  	  	#self.f_ABs = pd.DataFrame(columns=["fAB"])
		#self.delta_f_ABs = pd.DataFrame(columns=["deltafAB"])
		self.f_ABs = []
		self.delta_f_ABs = []
		
		self.f_ABs.append(self.f_of_A_B)
  
  	# Function that returns our current f_of_A_b evaluation
	def current_evaluation(self):
		return (0.5)*np.power(np.linalg.norm(self.y - np.dot(self.X, np.dot(self.A, self.B)), ord='fro'), 2) + self.lambda_1*np.sum(np.abs(self.A)) + self.lambda_2*np.sum(np.abs(self.B))
	
	# Build our model to predict.
	def fit(self):
        
		
		self.update_argmin_matrix_B()
		self.update_argmin_matrix_A()
		
		# Iterate until our f(A,B) is inside the epislon, Note that has_converged also updates the f_of_A_B value.
		while self.has_converged() == False:
			
			self.update_argmin_matrix_B()
			self.update_argmin_matrix_A()
            
		# When our function has convereged we have our final A, B values.
		A_rank = np.linalg.matrix_rank(self.A)
		B_rank = np.linalg.matrix_rank(self.B) # assuming this works with ndarray?

		# Not sure why we need to make this subtraction ******
		if A_rank < self.m or B_rank < self.m:
			self.m = self.m - 1
        
	# This method checks wether or not this iteration has found a f(A,B) such that it is appropriate for a model. ie, its inside epsilon
	def has_converged(self):
        
		self.f_ABs.append(self.f_of_A_B)
		
		# Our evaluation function is defined as: (1/2)||Y-XAB||F^2 (no lambdas as they are initially set to 0)
		f_of_A_B_new = (0.5)*np.power(np.linalg.norm(self.y - np.dot(self.X, np.dot(self.A, self.B)), ord='fro'), 2) + self.lambda_1*np.sum(np.abs(self.A)) + self.lambda_2*np.sum(np.abs(self.B))

		print "f_of_A_B_new"
		print f_of_A_B_new

		# Now finding our value to compare epsilon to, the amount of change in our cost function
		change_in_evaluation = abs(self.f_of_A_B - f_of_A_B_new) / self.f_of_A_B
		
		self.delta_f_ABs.append(change_in_evaluation)
		print "change_in_evaluation = "
		print change_in_evaluation

		# Update the f_of_A_B value
		self.f_of_A_B = f_of_A_B_new
        
		# Compare to see if we have converged to solution for our model
		if change_in_evaluation < self.epsilon:
			return True
		else:
			return False
        
    # In finding matrix B_i+1 we are going to execute a lasso problem for each station
	def update_argmin_matrix_B(self):
		
		# Need matrix XA for convex problem definition
		XA = self.X.dot(self.A)
		
		# Define variables needed for cvxpy
		Bv = cvx.Variable(self.m, self.num_stations)

		objective = cvx.Minimize( 0.5*cvx.sum_squares(self.y - XA*Bv) + self.lambda_1*cvx.norm(Bv, 1) )

		# Define the cvxpy problem then solve to update
		prob = cvx.Problem(objective)
		prob.solve(solver=cvx.SCS)

		if prob.status != cvx.OPTIMAL:
		        raise Exception("Solver did not converge!")

		# Update weighting matrix B
		self.B = Bv.value
		    
    # Here we just execute one lasso to update A
	def update_argmin_matrix_A(self):
		
		# Define variables needed for cvxpy
		Av = cvx.Variable(self.num_features, self.m)
		objective = cvx.Minimize( 0.5*cvx.sum_squares(self.y - self.X*Av*self.B) + self.lambda_1*cvx.norm(Av,1) )

		# Define the cvxpy problem then solve to update
		prob = cvx.Problem(objective)
		prob.solve(solver=cvx.SCS)

		if prob.status != cvx.OPTIMAL:
		        raise Exception("Solver did not converge!")

		# Update A with the solution
		self.A = Av.value
    
		
	# Method to predict based on new inputs
	def predict(self, X):
		# prediction = output = y = XAB
		return np.dot(self.X, np.dot(self.A, self.B))
	
	
	# Test methods individually
	def test(self):
		
		# print self.A
		# print "self.X"
		# print self.X
		# print "self.y"
		# print self.y
		# print "self.A"
		# print self.A
		# print "self.B"
		# print self.B
				
		print self.current_evaluation()
		
		self.update_argmin_matrix_B()

		# print "self.B after update"
		# print self.B
		
		print self.current_evaluation()

		self.update_argmin_matrix_A()
		
		# print "self.A after update"
		# print self.A
		
		print self.current_evaluation()
		
		# self.has_converged()
		#
		# self.update_argmin_matrix_B()
		#
		# print "self.B after update"
		# print self.B
		#
		# self.update_argmin_matrix_A()
		#
		# print "self.A after update"
		# print self.A
		#
		# self.has_converged()
		