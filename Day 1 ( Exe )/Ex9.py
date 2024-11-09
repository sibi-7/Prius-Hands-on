import pandas as pd
import matplotlib.pyplot as plt

csv_data=pd.read_csv("D:\csv\weather_data.csv")

def analyser(region):
    data=csv_data[csv_data['Location']==region]
    data=data.sort_values('Date_Time')
    data=data.reset_index()
    data=data.drop('index', axis=1)
    data=data.drop('Location',axis=1)
    data=data.set_index('Date_Time')
    return data
print_data=analyser('New York')
print(print_data)
