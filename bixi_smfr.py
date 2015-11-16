import pandas as pd
import sqlite3

# Read sqlite query results into a pandas DataFrame
con = sqlite3.connect("MontrealBikeData/mtl_bike_data_with_date.db")

# Choosing all dates that are from 2015, April=4, day= 19-25 
# df = pd.read_sql_query("SELECT * FROM data WHERE month = '4' AND day > '18' AND day < '26'", con)

# For each 5 minute interval in the week
day = 19
hour = 0
minute_lower_bound = 0
minute_upper_bound = 5

# Here we generate a query to get a specific 5 minute window of station data
query = "SELECT * FROM data WHERE month = '4' AND day = " + str(day) + " AND hour = " + str(hour) + " AND minute >= " + str(minute_lower_bound) + " AND minute < " + str(minute_upper_bound)
df = pd.read_sql_query(query, con)



print df