import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')
df = pd.read_csv('Iris.csv')
df.head()
df.shape
df.columns.tolist()
df['Species'].value_counts()
df = df.drop('Id', axis=1)
df.head()
df.isnull().sum()
num_duplicates = df.duplicated().sum()
print(f"Number of duplicated rows: {num_duplicates}")
df=df.drop_duplicates()
num_duplicates_rem = df.duplicated().sum()
print(f"Number of duplicated rows after removal: {num_duplicates_rem}")

df.info()
df.describe().T
sns.pairplot(df, hue='Species')
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
sns.set(style="whitegrid")

plt.figure(figsize=(12, 6)) 
sns.boxplot(x="Species", y="SepalLengthCm", data=df, palette="Set1")
plt.title("Box Plot of Sepal Length by Species")
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(x="Species", y="SepalWidthCm", data=df, palette="Set1")
plt.title("Box Plot of Sepal Width by Species")
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(x="Species", y="PetalLengthCm", data=df, palette="Set1")
plt.title("Box Plot of Petal Length by Species")
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(x="Species", y="PetalWidthCm", data=df, palette="Set1")
plt.title("Box Plot of Petal Width by Species")
plt.show()

from sklearn.preprocessing import LabelEncoder
cols=['Species']
le=LabelEncoder()
df[cols]=df[cols].apply(le.fit_transform)
df.head()

X = df.drop('Species', axis = 1)
X
y = df['Species']
y

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression(multi_class='multinomial', solver='lbfgs')
LR.fit(X_train, y_train)
LR_pred = LR.predict(X_test)

from sklearn.metrics import confusion_matrix,accuracy_score
confusion_matrix(y_test,LR_pred)
accuracy_score(y_test,LR_pred)
from sklearn.metrics import precision_score, recall_score
precision = precision_score(y_test, LR_pred, average='weighted')
recall = recall_score(y_test, LR_pred, average='weighted')
print("Precision:", precision)
print("Recall:", recall)

from sklearn.metrics import f1_score 
f1_score = f1_score(y_test, LR_pred, average='weighted')
print("F1 Score:", f1_score)
from sklearn.metrics import classification_report
report = classification_report(y_test, LR_pred)
print("Classification Report:\n", report)
from sklearn.tree import DecisionTreeClassifier
DT=DecisionTreeClassifier()
DT.fit(X_train,y_train)
DT_pred = DT.predict(X_test)

confusion_matrix(y_test,DT_pred)
accuracy_score(y_test,DT_pred)
report = classification_report(y_test, DT_pred)
print("Classification Report:\n", report)
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train,y_train)
rfc_pred = rfc.predict(X_test)

confusion_matrix(y_test,rfc_pred)
accuracy_score(y_test, rfc_pred)
report = classification_report(y_test, rfc_pred)
print("Classification Report:\n", report)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train,y_train)
knn_pred = knn.predict(X_test)

confusion_matrix(y_test,knn_pred)
accuracy_score(y_test,knn_pred)
report = classification_report(y_test, knn_pred)
print("Classification Report:\n", report)
