import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Load data
csv_data = pd.read_csv("D:/csv/weather_data.csv")

# Correct month and day labels
months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
weekend = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# Function to convert date to month, weekday, and day
def convert_date(x):
    date = x.split()[0]
    date = datetime.strptime(date, "%Y-%m-%d")
    return [date.month, date.isoweekday(), date.day]

# Function to add labels to bars
def bar_label(axes):
    for container in axes.containers:
        axes.bar_label(container, label_type="center", rotation=90)

# Plot function
def plots(df, name, num, axes, date=False):
    group = df.groupby(name)
    mean = group[num].mean()
    
    if not date:
        sns.barplot(x=mean.index, y=mean, ax=axes)
    else:
        mean = pd.DataFrame(mean)
        mean = mean.sort_index(ascending=True)
        dd = {num: [], "values": []}
        
        for i in range(mean.shape[0]):
            if name == "month":
                dd[num] += [months[mean.index[i] - 1]]
            else:
                dd[num] += [weekend[mean.index[i] - 1]]
            dd["values"] += [mean.iloc[i, 0]]
        
        dd = pd.DataFrame(dd)
        sns.barplot(x=dd.iloc[:, 0].values, y=dd.iloc[:, 1].values, ax=axes)

# Preprocess date information
csv_data = csv_data.sort_values("Date_Time")
csv_data["month"] = csv_data["Date_Time"].apply(lambda x: convert_date(x)[0])
csv_data["weekday"] = csv_data["Date_Time"].apply(lambda x: convert_date(x)[1])
csv_data["day"] = csv_data["Date_Time"].apply(lambda x: convert_date(x)[2])
csv_data["date"] = csv_data["Date_Time"].apply(lambda x: x.split()[0])

# Columns for plotting
nums = csv_data.columns[2:-4]

# Set up figure
fig, axes = plt.subplots(nrows=1, ncols=len(nums), figsize=(15, 7))

# Generate plots
for i, j in enumerate(nums):
    plots(csv_data, "month", j, axes[i], True)
    axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=90)
    bar_label(axes[i])
    axes[i].set_ylabel("")
    axes[i].set_xlabel(' '.join(j.split('_')[:-1]))

plt.tight_layout()
plt.show()
