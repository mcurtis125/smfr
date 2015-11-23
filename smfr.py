# Michael Curtis 260475694
# Sparse Multivariate Factor Regression
import numpy 	as np
import cvxpy 	as cvx
import pandas	as pd
from sklearn import linear_model


# First we will define a class for smfr to initialize needed variables for regression.
class SMFR:
	
	# Initialize inputs to local variables
	def __init__(self, X, y, lambda_1, lambda_2, m, epsilon):
		
		# X: n x p = num_data_points x num_features
		# y: n x q = num_data_points x num_stations
		# A: p x m = num_features x rank
		# B: m x q = rank x num_stations
		self.X = X
		self.y = y
		self.lambda_1 = lambda_1
		self.lambda_2 = lambda_2
		self.epsilon = epsilon
		self.m = m
		self.num_data_points = X.shape[0]
		self.num_features = X.shape[1]
		self.num_stations = y.shape[1]

		# Need to initialize our weight matrices A and B, A0, B0 will be random N(0,1).
		self.A = np.random.rand(self.num_features, self.m)
		self.B = np.random.rand(self.m, self.num_stations)
		
		# Initiallizing the evaluation value, this will always be the previous value		
		self.f_of_A_B = (0.5)*np.power(np.linalg.norm(self.y - np.dot(self.X, np.dot(self.A, self.B)), ord='fro'), 2) \
			 				+ self.lambda_1*np.sum(np.abs(self.A)) \
							+ self.lambda_2*np.sum(np.abs(self.B))		
  		
		# Needed data for visualization
		self.f_ABs = []
		self.delta_f_ABs = []
  
	# Build our model to predict.
	def fit(self):
        
		# This loop will be broken once we have eliminated all repeating linear combinations of rows of A and B, i.e. decreased m as much as possible
		while True:
		
			# With each iteration's change of the m value we need to initialize our weight matrices A and B, A0, B0 will be random N(0,1).
			self.A = np.random.rand(self.num_features, self.m)
			self.B = np.random.rand(self.m, self.num_stations)
		
			# Perform first update before doing a has_converged check
			self.update_argmin_matrix_B()
			self.update_argmin_matrix_A()
		
			# Iterate until our f(A,B) is inside the epislon, Note that has_converged also updates the f_of_A_B value.
			while self.has_converged() == False:
			
				self.update_argmin_matrix_B()
				self.update_argmin_matrix_A()
            
			# When our function has convereged we have our final A, B values.
			A_rank = np.linalg.matrix_rank(self.A)
			B_rank = np.linalg.matrix_rank(self.B)

			# check wether or not m is less than the rank of A and B final
			if  np.linalg.matrix_rank(self.A) < self.m or np.linalg.matrix_rank(self.B) < self.m:
				self.m = self.m - 1
			
			# We break the loop if we can't make m any smaller
			else:
				break
	
	# This method checks wether or not this iteration has found a f(A,B) such that it is appropriate for a model. ie, its inside epsilon
	def has_converged(self):
        
		# Add to our data for evaluations
		self.f_ABs.append(self.f_of_A_B)
		
		# Update our current Evaluation function
		f_of_A_B_new = (0.5)*np.power(np.linalg.norm(self.y - np.dot(self.X, np.dot(self.A, self.B)), ord='fro'), 2) \
						+ self.lambda_1*np.sum(np.abs(self.A)) \
						+ self.lambda_2*np.sum(np.abs(self.B))

		# Now finding our value to compare epsilon to, the amount of change in our cost function
		change_in_evaluation = np.abs(self.f_of_A_B - f_of_A_B_new) / self.f_of_A_B
		
		# Again add to our visualization data
		self.delta_f_ABs.append(change_in_evaluation)

		# Update the f_of_A_B value
		self.f_of_A_B = f_of_A_B_new
        
		# Compare to see if we have converged to solution for our model
		if change_in_evaluation < self.epsilon:
			return True
		else:
			return False
        
    # In finding matrix B_i+1 we are going to execute a lasso problem for each station
	def update_argmin_matrix_B(self):
		
		# Define variables needed for cvxpy and then define the problem with no constraints
		XA = self.X.dot(self.A)
		Bv = cvx.Variable(self.m, self.num_stations)
		objective = cvx.Minimize( 0.5*cvx.sum_squares(self.y - XA*Bv) + self.lambda_1*cvx.norm(Bv, 1) )
		prob = cvx.Problem(objective)
		
		# Solve the problem with SCS and make sure it converged
		prob.solve(solver=cvx.SCS)
		if prob.status != cvx.OPTIMAL:
		        raise Exception("Solver did not converge!")

		# Update weighting matrix B
		self.B = Bv.value
		    
    # Here we just execute one lasso to update A
	def update_argmin_matrix_A(self):
		
		# Define variables needed for cvxpy and then define the problem with no constraints
		Av = cvx.Variable(self.num_features, self.m)
		objective = cvx.Minimize( 0.5*cvx.sum_squares(self.y - self.X*Av*self.B) + self.lambda_1*cvx.norm(Av,1) )
		prob = cvx.Problem(objective)
		
		# Solve the problem with SCS and make sure it converged
		prob.solve(solver=cvx.SCS)
		if prob.status != cvx.OPTIMAL:
		        raise Exception("Solver did not converge!")

		# Update A with the solution
		self.A = Av.value
		
	# Method to predict based on new inputs. prediction = output = y = XAB
	def predict(self, X):
		return np.dot(self.X, np.dot(self.A, self.B))
	
  	# Function that returns our current f_of_A_b evaluation !!! NEED LAMBDAS???
	def current_evaluation(self):
		return (0.5)*np.power(np.linalg.norm(self.y - np.dot(self.X, np.dot(self.A, self.B)), ord='fro'), 2) \
			 	+ self.lambda_1*np.sum(np.abs(self.A)) \
				+ self.lambda_2*np.sum(np.abs(self.B))
		
		