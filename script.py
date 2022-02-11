import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

df = pd.read_csv("./asset/spam_ham_dataset.csv")
df.drop(["label"], axis=1, inplace=True)
df.rename({"label_num": "label", "text": "email"}, axis=1, inplace=True)

# Explore data
print(df.head())
print(df.info())
print(df.describe()) # it just shows the spam/ham ratio as indirect (not necessary)
print(df.isnull().sum())
df.dropna(inplace=True) # 1 row's email column was null before this block :)
print(df.isnull().sum()) # checking whether the code one row above works

df['label'].value_counts().plot(kind="barh", color=["r", "g"])
#plt.show()

# Prepare data for training
X = df["email"].values
y = df["label"].values

def remove_punctuations(data):
    for i in range(len(data)):
        data[i] = re.sub(r"[^\w\s]","", data[i])
        data[i] = re.sub("\s\s+", " ", data[i])
    return data

X = remove_punctuations(X)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

vectorizer = CountVectorizer(stop_words=ENGLISH_STOP_WORDS, max_df=0.9, min_df=0.0005)
x_train = vectorizer.fit_transform(x_train)
x_test = vectorizer.transform(x_test)

model = BernoulliNB()
model.fit(x_train, y_train)

preds = model.predict(x_test)
print(f1_score(y_test, preds))
print(precision_score(y_test, preds))
print(recall_score(y_test, preds))
print(confusion_matrix(y_test, preds))




