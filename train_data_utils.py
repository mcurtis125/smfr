import sqlite3
import os
import datetime
import sys
import numpy as np
import time

data_table_name = "data"
weather_table_name = "weather_data"

def gettraindata_x_ndarray():

	conn = sqlite3.connect('bikedb')
	c = conn.cursor()

	# start timer
	start_time = time.time()

	# SELECT ALL MONTHS AND DAYS
	c.execute("select distinct month from " + data_table_name + " order by month")
	months = list(sum(c.fetchall() , ()))
	c.execute("select distinct day from " + data_table_name + " order by day")
	days = list(sum(c.fetchall() , ()))
	# FOR TESTING: SELECT ONE WEEK
	# days = [17, 18, 19, 20, 21, 22, 23, 24]

	# get all months
	c.execute("select distinct hour from " + data_table_name + " order by hour")
	hours = list(sum(c.fetchall() , ()))

	# get number of stations
	c.execute("select count(distinct station) from " + data_table_name)
	num_stations = c.fetchone()[0]

	num_datapoints = len(months)*len(days)*len(hours)

	num_datetime_params = 4
	num_weather_params = 12

	num_columns = num_datetime_params+num_weather_params+num_stations

	result = np.ndarray(shape=(num_datapoints, num_columns), dtype=float)
	np.set_printoptions(threshold=num_datapoints*num_columns+1)
	index = 0
	c.execute("create view bike_changes as select month, day, hour, station, sum(increase) as sum_incr, sum(decrease) as sum_decr from " + data_table_name + " group by month, day, hour, station order by month, day, hour, station")
	for m in months:
		for d in days:
			for h in hours:
				c.execute("select sum_incr from bike_changes where month = ? and day=? and hour=?", (m,d,h))
				bikes_arrived = c.fetchall()
				c.execute("select sum_decr from bike_changes where month = ? and day=? and hour=?", (m,d,h))
				bikes_left = c.fetchall()
				c.execute("select month, dayofweek, day, hour, icon, precipType, precipIntensity, precipProbability, temperature, apparentTemperature, dewPoint, humidity, windSpeed, windBearing, cloudCover, pressure, ozone from " + weather_table_name + " where month = ? and day=? and hour=?", (m,d,h))
				weather = c.fetchall()
				arr = weather+bikes_arrived+bikes_left;
				arr = list(sum(arr, ()))
				np.put(result[index], np.arange(num_columns), arr)
				index += 1

	c.execute("drop view bike_changes")
	conn.close()
	# print result
	print("--- %s seconds ---" % (time.time() - start_time))
	return result
	# sys.exit()

# def gettraindata_y_ndarray():
