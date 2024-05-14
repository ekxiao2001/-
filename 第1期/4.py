import pandas as pd 
import numpy as np
from sklearn.linear_model import LinearRegression

# data init
train_data = pd.read_csv("songs_train.csv")
test_data = pd.read_csv("songs_test.csv")
X_train = train_data.drop("popularity", axis=1)
y_train = train_data["popularity"]
X_test = test_data.drop("popularity", axis=1)

# create model
model = LinearRegression()
model.fit(X_train, y_train)

# prediction
y_test = model.predict(X_test)

# save result
test_data["popularity"] = y_test
test_data.to_csv("songs_testout.csv", index=False)