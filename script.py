import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("./asset/spam_or_not_spam.csv")

# Explore data
print(df.head())
print(df.info())
print(df.describe()) # it just shows the spam/ham ratio as indirect (not necessary)
print(df.isnull().sum())
df.dropna(inplace=True) # 1 row's email column was null before this block :)
print(df.isnull().sum()) # checking whether the code one row above works

df['label'].value_counts().plot(kind="barh", color=["r", "g"])
plt.show()
