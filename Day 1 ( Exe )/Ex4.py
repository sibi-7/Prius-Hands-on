import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv(r"D:\csv\Electric_Vehicle_Population_Data.csv")
makes = data["Make"].value_counts().head(5)
makes.plot(kind='bar', color = ['red', 'green','blue','gray','orange'])
plt.title('manufacturer')
plt.show()
