import numpy as np
import smfr
import sqlite3

# Potential issues: 
# RUN WITH ONLY 10 STATIONS!!!!

# Main method to execute smfr for bixi data
def main():
	
	# Define our table and database names
	db_name = "bikedb"
	X_table_name = "x_data_hourly"
	y_table_name = "y_data_hourly"

	# Get the wanted station info for pulling from X and y
	stations = get_station_ids(db_name)
	X_string_columns, y_string_columns = get_query_station_strings(stations, 10)
	print "X_string_columns"
	print X_string_columns
	print "y_string_columns"
	print y_string_columns

	# Import the X and y sql tables as ndarrays
	X = get_sql_table_X(db_name, X_table_name, X_string_columns)
	y = get_sql_table_y(db_name, y_table_name, y_string_columns)
	
	# Now we need to split into testing and training data
	X_train, X_test = np.vsplit(X, [168])
	y_train, y_actual = np.vsplit(y, [168])
	print "y_train"
	print y_train.shape
	print "y_actual"
	print y_actual.shape
	print "X_train"
	print X_train.shape
	print "X_test"
	print X_test.shape
	
	# y_train = y
	# np.split(y, 2)
	#X_train = X[0]
	#print X_train
	# y_train = y[0]
	#X_test = X[1]
	#print X_train.shape
	#print X_test.shape
	# y_test_actual = y[1]
	# Now define our SMFR and make a prediction
	
	lambda_1 = 0.1
	lambda_2 = 0.1
	epsilon = 0.001
	m_initial = 12
	model = smfr.SMFR(X_train, y_train, lambda_1, lambda_2, m_initial, epsilon)
	model.fit()
	y_predicted = model.predict(X_test)
	print "y_actual"
 	print y_actual
	print "y_predicted"
	print y_predicted
	
	# Write A,B and y_tests output matrices to a csv
	np.savetxt("A.csv", model.A, delimiter=",")
	np.savetxt("B.csv", model.B, delimiter=",")
	np.savetxt("y_actual.csv", y_actual, delimiter=",")
	np.savetxt("y_predicted.csv", y_predicted, delimiter=",")
	
# Need a method to get a string to represent all of the column names for X and y queries
def	get_query_station_strings(stations, one_for_every):
	X_string_columns = ""
	y_string_columns = ""
	len_stations = len(stations)	
	for i, station in enumerate(stations):
		# We only want a subset of the number of stations so the data is not too big for cvxpy
		if i % one_for_every == 0:
			id = str(station[0])
			# if last iteration don't add a comma after the string
			if i + one_for_every > len_stations - 1:
				X_string_columns += "bikes_arrived_" + id + ", bikes_left_" + id + " "
				y_string_columns += "nbBikes" + id + ", nbEmptyDocks" + id + " "
			# otherwise do add that comma
			else:
				X_string_columns += "bikes_arrived_" + id + ", bikes_left_" + id + ", "
				y_string_columns += "nbBikes" + id + ", nbEmptyDocks" + id + ", "
	return X_string_columns, y_string_columns
		

# We need to get an array of all the possible stations
def get_station_ids(db_name):
	conn = sqlite3.connect(db_name)
	cur = conn.cursor()
	cur.execute("SELECT * FROM station_ids ORDER BY station")
	results = cur.fetchall()
	return np.asarray(results)
	
# We need to have a function that gets all the data form an sql table
def get_sql_table_X(db_name, table_name, X_string_stations):
	conn = sqlite3.connect(db_name)
	cur = conn.cursor()
	cur.execute("SELECT month, dayofweek, day, hour, icon, precipType, precipIntensity, precipProbability, \
			 	temperature, apparentTemperature, cloudCover, " + X_string_stations +  \
				" FROM " + table_name)
	results = cur.fetchall()
	return np.asarray(results)

def get_sql_table_y(db_name, table_name, y_string_stations):
	conn = sqlite3.connect(db_name)
	cur = conn.cursor()
	# Note: I removed the month, day, hour
	cur.execute("SELECT " + y_string_stations + " FROM " + table_name + " ORDER BY month, day, hour")
	results = cur.fetchall()
	return np.asarray(results)

# Main function
if __name__ == "__main__":
    main()