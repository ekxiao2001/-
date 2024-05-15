import pandas as pd 
import numpy as np

renet_data = pd.read_json('resnet_train.json')
inception_data = pd.read_json('inception_train.json')
xception_data = pd.read_json('xception_train.json')

# data pre-process
X = np.array([])
Y = np.array([])
data_size = len(renet_data.iloc[0])
for i in range(data_size):
    feature = renet_data.iloc[0,i] + inception_data.iloc[0,i] + xception_data.iloc[0,i]
    label = renet_data.iloc[1,i]
    X = np.append(X, np.array(feature))
    Y = np.append(Y, np.array(label))

X = np.reshape(X, (data_size,-1))
Y = np.reshape(Y, (data_size,-1))

split_factor = 0.7
X_train = X[:int(data_size*0.7), :]
Y_train = Y[:int(data_size*0.7), :]
X_test = X[int(data_size*0.7):, :]
Y_test = Y[int(data_size*0.7):, :]

# model train
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, Y_train)
model.predict(X_test)
model.score(X_test, Y_test)

# prediction and save result
test_data = pd.read_json('test.json')

test = np.array([])
test_data_size = len(test_data.iloc[0])
for i in range(test_data_size):
    feature = test_data.iloc[0,i]
    test = np.append(test, np.array(feature))

test = np.reshape(test, (test_data_size,-1))

pred = model.predict(test)

with open('result.csv', 'w') as g:
    g.write('id,label\n')
    for i in range(len(pred)):
        g.write(test_data.columns[i] + ',' + str(int(pred[i])) + '\n')