import pandas as pd
data=pd.read_csv("D:\csv\Electric_Vehicle_Population_Data.csv")
print(data["Make"].value_counts())
