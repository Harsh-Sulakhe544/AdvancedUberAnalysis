BIGGEST DRAWBACK FOR THIS JUPYTER-NOTEBOOK IS , EACH TIME U OPEN VS-CODE , U NEED TO CLEAR ALL O/P'S => THEN 
RE-RUN THE CELLS FROM BEGINNING  => cuz IMAGES TAKES UP A LOT OF DATA 

WHILE QUITING ALSO ALWAYS CLEAR-ALL-O/P'S AND THEN SAVE AND THEN CLOSE 

ALWAYS USE TEST-DATA => FOR PROPER UNDERSTANDING (FIRST 100 ROWS OR 50 ROWS )

HERE ONCE-DATASET.CSV => CREATED => NO REPEATED  


what are DataFrames in Python? 

+----------------------------------+
|           Pandas DataFrame      |
+----------------------------------+
|           DataFrame Name        |
+----------------------------------+
|              Metadata            |
+----------------------------------+
|               Index              |
+----------------------------------+
|              Columns             |
|   +------+--------+--------+     |
|   | Col1 | Col2   | Col3   |     |
|   +------+--------+--------+     |
|   |  ... |  ...   |  ...   |     |
|   +------+--------+--------+     |
+----------------------------------+
|               Data               |
|   +------+------+------+         |
|   |  1   |  2   |  3   |         |
|   +------+------+------+         |
|   |  ... |  ... |  ... |         |
|   +------+------+------+         |
+----------------------------------+


how is it linked with python 

+----------------------------------+
|           Python Interpreter    |
+----------------------------------+
|            Your Python Code      |
+----------------------------------+
|           Pandas Library         |
+----------------------------------+
|         NumPy and Other          |
|         External Libraries       |
+----------------------------------+
|           Operating System       |
+----------------------------------+

DataFrames in Python are stored in the main memory of the computer system. This is because DataFrames are 
essentially just NumPy arrays with an extra layer of indexing and metadata. As such, they take up the same 
amount of memory as a NumPy array of the same size.

STEPS FOR THIS PROJECT : 
1- Import necessary libraries
2- Read the dataset using Pandas => uber-raw-data-sep14.csv
3- Explore the dataset properties
4- Visualize the relationship between different variables and draw insights => heatmap / 
cluser-clustered_map_with_tooltips
5. consider one more dataset => weather.csv , which has details like rainfall windfall etc , 
6.combine the weather.csv with uber-raw-data-sep14.csv to peform analysis of rainfall and plot the rainfall 
trends wrt date and amount of rainfall => merged_data.csv

7.now Select the keys of merged_data.csv  => 'Lat', 'Lon', 'MinTemp', 'MaxTemp', 'Rainfall' na perform 
the deep analysis to get location with more riders  
8.finally create a exp_merge_data.csv => 'Lat', 'Lon', 'MinTemp', 'MaxTemp', 'Rainfall'  + 'DefaultPrice' and perform PROFIT ANALYSIS on whole above dataset exp_merge_data.csv

FINAL O/P => IN HEATMAP_WITH_PRICED_ANALYSIS.html =>  open file in chrome , click on geo-symbol => to get 
price also 
Green: Low temperatures or minimal rainfall.
Yellow: Moderate temperatures or moderate rainfall.
Orange: High temperatures or increased rainfall.
Red: Very high temperatures or intense rainfall.

MODULES USED 

pandas => to read the dataset ,
matplotlib.pyplot as plt => to vsualize dataset in insightful way 
import folium
from folium.plugins import HeatMap => ZOOMED MAP 
from folium.plugins import MarkerCluster => MAP SIMILAR TO GOOGLE-MAPS
numpy  => for exp_meged_data.csv file

built-in methods for pandas and matplotlib 

pd.read_csv('path/to/.csv') => to read the csv file using pandas(pd)
head(5) = #Display the first 5 records of .csv file
tail() = #Display the last 5 records of .csv file

shape => #Find the shape of the dataset (check the number of rows and columns )

info() = #Understand the dataset properties , like detail about each column ex : 
# Column 1 (Lat) has 1028136 non-null entries of type float64.
# Column 2 (Lon) has 1028136 non-null entries of type float64

pd.to_datetime(object['column-name']) , remember column-name = key to access

to get details from previous columns => use apply()

apply(lambda x: x.day) => LAMBDA IS SLOWER TO EXECUTE 
apply(lambda x: x.hour)
apply(lambda x: x.weekday())

#Visualize the Density of rides per Day of month, creates a new figure and axes for the upcoming histogram 
plot. The figsize , fig=> return object , ax => array of x-y-axis to interact in future with histogram to get 
figure 
fig,ax = plt.subplots(figsize = (12,6))

# plot the histogram 
plt.hist(uber_df.Day, width= 0.6, bins= 30) , 30 means => 30 bars

# fetch the latitiude and longitude of each iuber rider
x= uber_df.Lon 
y= uber_df.Lat

plt.scatter(x, y, color= "yellow") # scatter based on latitude and longitude for each uber-rider 

plt.figure(figsize=(12, 6)) => Figure size 1200x600 with 0 Axes>  => creates a object 

# Adding Title and Labels to histogram 
plt.title("Density of trips per Day", fontsize=16) = title of graph 
plt.xlabel("Day", fontsize=14)  = xaxis label 
plt.ylabel("Density of rides", fontsize=14) = yaxis label 

plt.grid() => to create a grid (small notebook lines square like graph)

plt.plot() => The plot() function is used to draw points (markers) in a diagram. By default, the plot() 
function draws a line from point to point.  

plt.show() => it starts an event loop, looks for all currently active figure objects, and opens one or more 
interactive windows that display your figure or figures.

plt.legend() => The elements to be added to the legend are automatically determined, when you do not pass in 
any extra arguments

Marker() => this function in Python is used to create a marker object. A marker object is a graphical object 
that can be used to represent data points on a plot. 

to_csv() => it is a built-in function in Pandas that allows you to save a Pandas DataFrame as a CSV file.
folium.Map() => to create a folium map
describe() => to get complete info  statistics 
merge() => This method updates the content of two DataFrame by merging them together,

iterrows() => this  method generates an iterator object of the DataFrame, allowing us to iterate each row in 
the DataFrame.Each iteration produces an index object and a row object (a Pandas Series object).

dropna() -> pandas => The dropna() method removes the rows that contains NULL values.
The dropna() method returns a new DataFrame object unless the inplace parameter is set to True, in that case 
the dropna() method does the removing in the original DataFrame instead.

add_to() => this method is used to add a GeoJSON object to a map. It takes the map object as an argument and 
returns the GeoJSON object

folium.LayerControl => This will create a map with a layer control that allows the user to toggle between 
different layers.



BACKUP CODE -- use clear-all-o/p's options  => and run from START , reached 13 => 
open pickup_heatmap.html => with chrome 

1. #To read the dataset
import pandas as pd 

#For visualization
import matplotlib.pyplot as plt

2.#Read the dataset
uber_df= pd.read_csv("Data/uber-raw-data-sep14.csv")

#Display the first 5 records
uber_df.head(5)

3.#Display the last 5 records
uber_df.tail()

4.uber_df.shape

5.#Understand the dataset properties , like detail about each column ex : 
# Column 1 (Lat) has 1028136 non-null entries of type float64.
# Column 2 (Lon) has 1028136 non-null entries of type float64
uber_df.info()
6 => this will min take 1 MIN - 2MIn or 3 MINs to execute
6.#Change the "Date/Time" column's data type from string to datetime, so that we get realistic info
uber_df['Date/Time']= pd.to_datetime(uber_df['Date/Time'])

#Convert "Date/Time" column from string data type into DateTime, use 
# The apply method calls lambda function, and applies the computation to each row of the data frame. Besides, apply 
# can also do the modification for every column in the data frame., 
# extract day , month , weekday using Date/Time column 

uber_df["Day"] = uber_df["Date/Time"].apply(lambda x: x.day)
uber_df["Hour"] = uber_df["Date/Time"].apply(lambda x: x.hour)
uber_df["Weekday"] = uber_df["Date/Time"].apply(lambda x: x.weekday())
uber_df.head(5)

7.#Visualize the Density of rides per Day of month, creates a new figure and axes for the upcoming histogram plot. The figsize , fig=> return object , ax => array of x-y-axis to interact in future with histogram to get figure 
fig,ax = plt.subplots(figsize = (12,6))
# plot the histogram 
plt.hist(uber_df.Day, width= 0.6, bins= 30)
# Adding Title and Labels to histogram 
plt.title("Density of trips per Day", fontsize=16)
plt.xlabel("Day", fontsize=14)
plt.ylabel("Density of rides", fontsize=14)

8.#Visualize the Density of rides per Weekday
fig,ax = plt.subplots(figsize = (12,6)) # 12 inches width and 6 inches height 
plt.hist(uber_df.Weekday, width= 0.6, range= (0, 6.5), bins=7, color= "red") #0.6->width of each bar in histo , 6.5 => include 6 
# also , 7=> intervals on x axis (mon-sun)
plt.title("Density of trips per Weekday", fontsize=16)
plt.xlabel("Weekday", fontsize=14) # 14 => 14/72 inches for text-size of x-axis 
plt.ylabel("Density of rides", fontsize=14) # 14 => 14/72 inches for text-size of y-axis


9.#Visualize the Density of rides per hour , cuz we see Saturday(x-axes = 6) is lazy , but Moday(x-axes=1) is Busy
fig,ax = plt.subplots(figsize = (12,6))
plt.hist(uber_df.Hour, width= 0.6, bins=24, color= "brown") # 24 => 24 hours in each day 
plt.title("Density of trips per Hour", fontsize=16)
plt.xlabel("Hour", fontsize=14)
plt.ylabel("Density of rides", fontsize=14)

10.# from the above diagram after 0,  1am-4am decreasing graph , then 5am-6pm (max-value) increasing (not strictly , but 
# profit for business ) 

#Visualize the Density of rides per location
fig,ax = plt.subplots(figsize = (12,6))
# fetch the latitiude and longitude 
x= uber_df.Lon 
y= uber_df.Lat
plt.scatter(x, y, color= "yellow") # scatter based on latitude and longitude for each uber-rider 
plt.title("Density of trips per Hour", fontsize=16)
plt.xlabel("Hour", fontsize=14)
plt.ylabel("Density of rides", fontsize=14)

11 - 13 => heatmap visualization for density of pickup or drop-off locations on the map.

11.import folium
from folium.plugins import HeatMap

12.# Create a base map centered at the average latitude and longitude (so that we can get the over-all view of dataset)
average_lat = uber_df['Lat'].mean()
average_lon = uber_df['Lon'].mean()

# find the location using latitude and longitude 
# we want to see distance in km => control_scale=True , zoom_start=11 => startingly display of map 
base_map = folium.Map(location=[average_lat, average_lon], control_scale=True, zoom_start=11)

13 -> this will take 1 - 3 mins or 10 mins maximum to execute 
13.# Plot pickup locations as HeatMap
# convert the numpy-array to standard list , 

# radius = 10 ? ->  sets the radius of influence of each data point on the heatmap. A larger radius means the 
# influence of each point will be spread over a larger area, resulting in a smoother heatmap.

# max_zoom = 13 -> beyond which heatmap is not proper -> 13 is best detailed view of heatmap

# .add_to(base_map): This method is used to add the heatmap layer to the base_map created earlier. The HeatMap object 
# is created with the specified parameters (pickup_heatmap_data, radius, max_zoom), and then it's added as a layer to 
# the base_map using the add_to method

pickup_heatmap_data = uber_df[['Lat', 'Lon']].values.tolist()
HeatMap(pickup_heatmap_data, radius=10, max_zoom=13).add_to(base_map)

# Display the map
base_map.save('pickup_heatmap.html')

14-> 16 best way to use for realworld => folium 
14 - 16 -> The density of pickup or drop-off locations on the map visualizes the concentration of Uber 
activities in different geographic areas. This information is represented by the intensity of color or "heat" 
on the map. Here's what you can infer from the density visualization:

Hotspots , Coldspots(higher and lower concentration of uber's Activity )
Patterns over Time, Geographic Trends (time specific wrt heatmap, popular routes )

14.import folium
from folium.plugins import MarkerCluster

15.average_lat = uber_df['Lat'].mean()
average_lon = uber_df['Lon'].mean()
base_map = folium.Map(location=[average_lat, average_lon], control_scale=True, zoom_start=11)

16.# Create a MarkerCluster for faster creation %time
%time
marker_cluster = MarkerCluster().add_to(base_map)

# Use a smaller subset of the data (e.g., first 100 rows) for testing
subset_df = uber_df.head(100)

# Iterate through the DataFrame and add markers to the cluster
for index, row in subset_df.iterrows():
    
    folium.Marker(
        location=[row['Lat'], row['Lon']],
        tooltip=f"Pickup time: {row['Date/Time']}",
    ).add_to(marker_cluster)

# Save the map
base_map.save('clustered_map_with_tooltips.html')

17.# i want to combine both the datasets to future analyze wrt rainfall for each lat and lon columns 
# Load weather.csv
weather_df = pd.read_csv("Data/weather.csv")

# Load uber_data.csv
uber_df = pd.read_csv("Data/uber-raw-data-sep14.csv")

18.# Select the first 50 rows from each dataframe
weather_subset = weather_df.head(50)
uber_subset = uber_df.head(50)

19.# Merge the dataframes based on an appropriate column (e.g., Date/Time)
merged_df = pd.merge(weather_subset, uber_subset, left_index=True, right_index=True)

20.# Save the merged dataframe to a new CSV file
merged_df.to_csv("Data/merged_data.csv", index=False)
# Display the merged dataframe
print(merged_df)

21.# Perform additional analysis as needed
# For example, you can use describe() to get descriptive statistics
analysis_result = merged_df.describe()
print(analysis_result)

# now from this merged-file we will extract day month year 
22.# Load the merged CSV file into a DataFrame
file_path = 'Data/merged_data.csv'
df = pd.read_csv(file_path)

23.# Convert "Date/Time" column to datetime format
df['Date/Time'] = pd.to_datetime(df['Date/Time'])

24.# Extract day, month, and year and create new columns
df['Day'] = df['Date/Time'].dt.day
df['Month'] = df['Date/Time'].dt.month
df['Year'] = df['Date/Time'].dt.year

25.#  Plot the trends of temperature over time
plt.figure(figsize=(12, 6))

temperature_column = 'MaxTemp'

plt.plot(df['Date/Time'], df[temperature_column], marker='o', label=temperature_column)
plt.title(f'Temperature Trends Over Time ({temperature_column})')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.grid(True)
plt.show()

26.# Plot the trends of rainfall over time
plt.figure(figsize=(12, 6))
# Assuming you have a 'Rainfall' column
rainfall_column = 'Rainfall'

plt.plot(df['Date/Time'], df[rainfall_column], marker='o', color='r', label=rainfall_column)
plt.title(f'Rainfall Trends Over Time ({rainfall_column})')
plt.xlabel('Date')
plt.ylabel('Rainfall (mm)')
plt.legend()
plt.grid(True)
plt.show()

27.# todo windfall and rainfall analysis , and then plot the heatmap for both in one single map
data = pd.read_csv('Data/merged_data.csv')

28.# Create a heatmap based on MinTemp, MaxTemp, and Rainfall, but remove the rows that contain null values 
using dropna()
locations = data[['Lat', 'Lon', 'MinTemp', 'MaxTemp', 'Rainfall']].dropna()

29.# Normalize the data for better visualization
locations['MinTemp'] = (locations['MinTemp'] - locations['MinTemp'].min()) / (locations['MinTemp'].max() - 
locations['MinTemp'].min())
locations['MaxTemp'] = (locations['MaxTemp'] - locations['MaxTemp'].min()) / (locations['MaxTemp'].max() - 
locations['MaxTemp'].min())
locations['Rainfall'] = (locations['Rainfall'] - locations['Rainfall'].min()) / (locations['Rainfall'].max() 
- locations['Rainfall'].min())

30.# Create a map centered around the mean of latitude and longitude
m = folium.Map(location=[locations['Lat'].mean(), locations['Lon'].mean()], zoom_start=10)

31.# Add heatmap layer with custom colors for each attribute
gradient = {0.2: 'blue', 0.4: 'green', 0.6: 'yellow', 0.8: 'orange', 1: 'red'}

HeatMap(locations[['Lat', 'Lon', 'MinTemp']].values, radius=15, gradient=gradient, name='MinTemp').add_to(m)
HeatMap(locations[['Lat', 'Lon', 'MaxTemp']].values, radius=15, gradient=gradient, name='MaxTemp').add_to(m)
HeatMap(locations[['Lat', 'Lon', 'Rainfall']].values, radius=15, gradient=gradient, name='Rainfall').add_to(m)

32.# Add layer control to toggle between different attributes
# Add popups for exact latitude and longitude in heatmap - to get precised location 
for _, row in locations.iterrows():
    folium.Marker([row['Lat'], row['Lon']],
                  popup=f"Lat: {row['Lat']}, Lon: {row['Lon']}").add_to(m)
folium.LayerControl(collapsed=False).add_to(m)

# Save the map as an HTML file
m.save('heatmap_with_detailed_analysis.html')

PRICE ANALYSIS 

33.# now add the random columnn DefaultPrice 
import numpy as np

34.# Load your existing merged_data.csv file
data = pd.read_csv('Data/exp_merged_data.csv')

35.# Add a new column 'DefaultPrice' with random values between $10 and $15
data['DefaultPrice'] = np.random.uniform(10, 15, len(data))
# Save the modified DataFrame back to the CSV file , dont-add the index-value  , so index=false
data.to_csv('Data/exp_merged_data.csv', index=False)

36.# now perform the price analysis for each area 
# Create a heatmap based on MinTemp, MaxTemp, Rainfall, and DefaultPrice
heatmap_data = data[['Lat', 'Lon', 'MinTemp', 'MaxTemp', 'Rainfall', 'DefaultPrice']].dropna()

37.# Normalize the data for better visualization
for column in ['MinTemp', 'MaxTemp', 'Rainfall', 'DefaultPrice']:
    heatmap_data[column] = (heatmap_data[column] - heatmap_data[column].min()) / (heatmap_data[column].max() - heatmap_data[column].min())
    
# Normalize the DefaultPrice column to fall between $10 and $15
heatmap_data['DefaultPrice'] = 10 + heatmap_data['DefaultPrice'] * 5   

38.# Create a map centered around the mean of latitude and longitude
m = folium.Map(location=[heatmap_data['Lat'].mean(), heatmap_data['Lon'].mean()], zoom_start=10)

39.# Add heatmap layer with custom colors for each attribute
gradient = {0.2: 'blue', 0.4: 'green', 0.6: 'yellow', 0.8: 'orange', 1: 'red'}

HeatMap(heatmap_data[['Lat', 'Lon', 'MinTemp']].values, radius=15, gradient=gradient, name='MinTemp').add_to(m)
HeatMap(heatmap_data[['Lat', 'Lon', 'MaxTemp']].values, radius=15, gradient=gradient, name='MaxTemp').add_to(m)
HeatMap(heatmap_data[['Lat', 'Lon', 'Rainfall']].values, radius=15, gradient=gradient, name='Rainfall').add_to(m)
HeatMap(heatmap_data[['Lat', 'Lon', 'DefaultPrice']].values, radius=15, gradient=gradient, name='DefaultPrice').add_to(m)

40.# Add popups for exact latitude, longitude, and price in heatmap - to get precise location and price
for _, row in heatmap_data.iterrows():
    folium.Marker([row['Lat'], row['Lon']],
                  popup=f"Lat: {row['Lat']}, Lon: {row['Lon']}, Price: ${row['DefaultPrice']:.2f}").add_to(m)

41.# Add layer control to toggle between different attributes
folium.LayerControl(collapsed=False).add_to(m)

# Save the map as an HTML file
m.save('heatmap_with_priced_analysis.html')

42.# it will imagine - not actually go and change the exp_merged_data.csv and insert the columns Revenue , Expense ,Profit
# Calculate revenue based on DefaultPrice
heatmap_data['Revenue'] = heatmap_data['DefaultPrice']

# Estimate expenses 
heatmap_data['Expenses'] = 0.2 * heatmap_data['Revenue']

# Calculate profit
heatmap_data['Profit'] = heatmap_data['Revenue'] - heatmap_data['Expenses']

43.# Sum up the total revenue, expenses, and profit
total_revenue = heatmap_data['Revenue'].sum()
total_expenses = heatmap_data['Expenses'].sum()
total_profit = heatmap_data['Profit'].sum()

# Print the results, 
# The \033[1m is an ANSI escape code that represents the beginning of a control sequence to change the text style. Specifically, \033[1m is used to set the text to bold, [0m => END 
print("\033[1mUBER ANALYSIS FOR PROFIT\033[0m")
print(f'Total Revenue: ${total_revenue:.2f}')
print(f'Total Expenses: ${total_expenses:.2f}')
print(f'Total Profit: ${total_profit:.2f}')
