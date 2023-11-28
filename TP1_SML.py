import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler

features_path = "acsincome_ca_features.csv"
label_path = "acsincome_ca_labels.csv"

X = pd.read_csv(features_path, index_col=False, sep=",")
y = pd.read_csv(label_path, index_col = False, sep=",")

# only use the first N samples to limit training time
num_samples = int(len(X)*0.1)
X, y = X[:num_samples], y[:num_samples]
# Standrize the data
scaler = StandardScaler()
scaler.fit_transform(X)
# Split the data
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

model = SVR()
model.fit(X_train,y_train) 
print("train score : ", model.score(X_train,y_train))
print("test score : ", model.score(X_test,y_test))

cross_val_score(SVR(), X_train,y_train,cv=5,scoring='accuracy')





