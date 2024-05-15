# 1. 读取三个json数据，并将feature字段合并
# 2. 使用处理好的数据训练一个分类模型
# 3. 将预测结果保存在“result.csv”文件中

import pandas as pd
import numpy as np
data_resnet = pd.read_json('resnet_train.json')
data_inception = pd.read_json('inception_train.json')
data_xception = pd.read_json('xception_train.json')
data_test = pd.read_json('test.json')

row_num = data_resnet.shape[1]
col_num = np.array(data_resnet.iloc[0,0]).shape[0]
X_resnet = np.zeros([row_num, col_num])
X_inception = np.zeros([row_num, col_num])
X_xception = np.zeros([row_num, col_num])
y = np.zeros([row_num, ])
X_test = np.zeros([data_test.shape[1], col_num*3])

for i in range(row_num):
    X_resnet[i] = np.array(data_resnet.iloc[0,i])
    X_inception[i] = np.array(data_inception.iloc[0,i])
    X_xception[i] = np.array(data_xception.iloc[0,i])
    y[i] = np.array(data_resnet.iloc[1,i])

for i in range(data_test.shape[1]):
    X_test[i] = np.array(data_test.iloc[0,i])

X = np.concatenate([X_resnet, X_inception, X_xception], axis=1)

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
score = model.score(X_val, y_val)
print(f"分类模型的模拟准确率为：{score}")

pred = model.predict(X_test)
with open('result.csv', 'w') as f:
    f.write("id,label\n")
    for i in range(pred.shape[0]):
        id = data_test.columns[i]
        f.write(id + ',' + f'{int(pred[i])}\n')
