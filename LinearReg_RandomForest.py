"""
This is project 3 of RICE Data Analytics and Vizualizations bootcamps
"""


# PART 1 - Data gathering and preparation

import pandas as pd
import numpy as np
import math
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Read columns names
url = "https://testbucketirina.s3.eu-west-2.amazonaws.com/data/features.txt"
columns_df = pd.read_csv(url, header=None)
columns = columns_df[0]

# Remove extra spaces from the columns' names
columns_clean = []
for i in range(len(columns)):
    column_name = columns[i]
    split_name = column_name.split(" ")[0]
    columns_clean.append(split_name)

# Read Test csv into df and add header
url_X_train = "https://testbucketirina.s3.eu-west-2.amazonaws.com/data/Train/X_train.txt"
X_train = pd.read_csv(url_X_train, sep='\s+', header=None)
X_train.columns = columns_clean

# Read target data
labels_url = "https://testbucketirina.s3.eu-west-2.amazonaws.com/data/activity_labels.txt"
labels = pd.read_csv(labels_url, header=None)

# Read target data and rename the column
y_train_url = "https://testbucketirina.s3.eu-west-2.amazonaws.com/data/Train/y_train.txt"
y_train = pd.read_csv(y_train_url, sep='\s+', header=None)
y_train = y_train.rename(columns={0: "Activity"})

# Merge X and y data into one df for correlation matrix
df = y_train.join(X_train, how='outer')

# read to df from S3 aws:
url_master = "https://testbucketirina.s3.eu-west-2.amazonaws.com/data/master.csv"
drive_s3_df = pd.read_csv(url_master)
drive_s3_df.head()

# PART 2 - Split and train the model

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

exp_df = df.copy()

# Import random forest classifier
from sklearn.ensemble import RandomForestClassifier

# Convert "Activity" column to numeric
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
exp_df['Activity'] = le.fit_transform(exp_df['Activity'].astype(str))

# Create a random forest classifier
clf = RandomForestClassifier(n_estimators=200)

# Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)

url_X_test = "https://testbucketirina.s3.eu-west-2.amazonaws.com/data/Test/X_test.txt"
X_test = pd.read_csv(url_X_test, sep='\s+', header=None)
X_test.columns = columns_clean

# Run prediction
y_pred=clf.predict(X_test)

# Print
print(y_pred)

url_y_test = "https://testbucketirina.s3.eu-west-2.amazonaws.com/data/Test/y_test.txt"
y_test = pd.read_csv(url_y_test, sep='\s+', header=None)
y_test = y_test.rename(columns={0: "Activity"})
y_test

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

# Random Forests in sklearn will automatically calculate feature importance
importances = clf.feature_importances_
print(f"Importances: {importances[:20]}")

# We can sort the features by their importance
print((sorted(zip(clf.feature_importances_, exp_df.columns), reverse=True))[:20])

feat_df = exp_df[['tGravityAcc-SMA-1', 'tGravityAcc-Min-1', 'tGravityAcc-Max-1',
                  'tBodyGyroJerk-AngleWRTGravity-1', 'tXAxisAcc-AngleWRTGravity-1',
                  'tGravityAcc-Mean-1', 'tGravityAcc-Mean-2', 'tBodyAcc-Correlation-3', 'tGravityAcc-Mad-3',
                  'tGravityAcc-Energy-1', 'tBodyAccJerk-STD-3', 'tGravityAccMag-ARCoeff-4',
                  'tGravityAcc-Max-2', 'fBodyAccMag-Mean-1', 'tBodyAccJerkMag-Min-1',
                  'Activity',]]
feat_df.head()

# PART 3 - Pre-processing after feature selection

join_test_df = y_test.join(X_test, how='outer')
join_train_df = y_train.join(X_train, how='outer')

combined_df = pd.concat([join_test_df, join_train_df])

from sklearn.model_selection import train_test_split

X = combined_df.drop("Activity", axis=1)
y = combined_df['Activity'].values.reshape(-1, 1)

print(f"PART3 - X.shape: {X.shape}, y.shape: {y.shape}")

data = X.copy()

# Split the dataset back into Train and Test
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Scale the data before training

from sklearn.preprocessing import StandardScaler
X_scaler = StandardScaler().fit(X_train)
y_scaler = StandardScaler().fit(y_train)

X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)
y_train_scaled = y_scaler.transform(y_train)
y_test_scaled = y_scaler.transform(y_test)

# Restart with scaled data
X = combined_df.drop("Activity", axis=1)
y = combined_df['Activity'].values.reshape(-1, 1)

# Split data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Scale or Normalize your data. Use StandardScaler if you don't know anything about your data.
from sklearn.preprocessing import StandardScaler
X_scaler = StandardScaler().fit(X_train)
y_scaler = StandardScaler().fit(y_train)

X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)
y_train_scaled = y_scaler.transform(y_train)
y_test_scaled = y_scaler.transform(y_test)

# Fit the Model to the scaled training data and make predictions using the scaled test data
# Plot the results 
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train_scaled, y_train_scaled)

# Quantify your model using the scaled data
from sklearn.metrics import mean_squared_error

predictions = model.predict(X_test_scaled)
MSE = mean_squared_error(y_test_scaled, predictions)
r2 = model.score(X_test_scaled, y_test_scaled)

print(f"Standard Scaler - MSE: {MSE}, R2: {r2}")

# PART 4 - MinMaxScaler

from sklearn.preprocessing import MinMaxScaler
X_scaler = MinMaxScaler().fit(X_train)
y_scaler = MinMaxScaler().fit(y_train)

X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)
y_train_scaled = y_scaler.transform(y_train)
y_test_scaled = y_scaler.transform(y_test)

# Step 1) Convert Categorical data to numbers using Integer or Binary Encoding
X = combined_df.drop("Activity", axis=1)
y = combined_df["Activity"].values.reshape(-1, 1)

# Step 2) Split data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Step 3) Scale or Normalize your data. Use MinMaxScaler if you don't know anything about your data.
X_scaler = MinMaxScaler().fit(X_train)
y_scaler = MinMaxScaler().fit(y_train)

X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)
y_train_scaled = y_scaler.transform(y_train)
y_test_scaled = y_scaler.transform(y_test)

# Step 4) Fit the Model to the scaled training data and make predictions using the scaled test data
# Plot the results 
model = LinearRegression()
model.fit(X_train_scaled, y_train_scaled)

# Step 5) Quantify your model using the scaled data
predictions = model.predict(X_test_scaled)
MSE = mean_squared_error(y_test_scaled, predictions)
r2 = model.score(X_test_scaled, y_test_scaled)

print(f"MSE: {MSE}, R2: {r2}")

print("Note: not much difference between StandardScaler and MinMaxScaler")














