import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv("D:\csv\Electric_Vehicle_Population_Data.csv")
print(data.columns)
makes=data["County"].value_counts().head(5)
makes.plot(kind='bar', color=['red','green', 'blue', 'gray', 'orange'])
plt.title('As per country')
plt.show()
