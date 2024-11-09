import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv("D:\csv\Electric_Vehicle_Population_Data.csv")

Data=data['Clean Alternative Fuel Vehicle (CAFV) Eligibility']
value_counts=Data.value_counts()
colors=['blue', 'red', 'gold', 'coral']

plt.figure(figsize=(6, 6))
plt.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%', colors=colors, startangle=140)

plt.title("eligibility for clean alternative fuels")
plt.axis("equal")
plt.show()
