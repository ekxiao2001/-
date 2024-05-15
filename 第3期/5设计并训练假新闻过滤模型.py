# 1. 读取txt文件 训练集，测试集，标签
# 2. 将训练集和测试集中的文本向量化
# 3. 构建分类模型进行训练
# 4. 使用测试集输出结果， 保存在 “pred_test.txt”文件中
# 5. 准确率不低于0.9（使用训练集合中的数据模拟指标）

import numpy as np

# 1. 读取txt文件 训练集，测试集，标签
with open("label_newstrain.txt", 'r') as f:
    labels = []
    for label in f:
        labels.append(int(label.strip()))
    labels = np.array(labels)

with open("news_train.txt", 'r') as f:
    text_train = []
    for text in f:
        text_train.append(text.strip())      
    text_train = np.array(text_train)

with open("news_test.txt", 'r') as f:
    text_test = []
    for text in f:
        text_test.append(text.strip())
    text_test = np.array(text_test)

# 2. 将训练集和测试集中的文本向量化
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
text = np.concatenate([text_train, text_test])
# CountVectorizer 方法会因输入文本的不同而产生不同维度的向量表示
# 因此这里将训练集和测试集合并，从而使训练集和测试集的向量表示维度一致，便于后续模型建立
X = vectorizer.fit_transform(text)[:text_train.shape[0], :]
X_test = vectorizer.fit_transform(text)[text_train.shape[0]:, :]
y = labels

# 3. 模型训练
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=1)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
score = model.score(X_val, y_val)
print(f"模型近似准确率为：{score}")

# 4. 使用测试集输出结果， 保存在 “pred_test.txt”文件中
pred = model.predict(X_test)
with open("pred_test.txt", 'w') as f:
    for i in range(pred.shape[0]):
        f.write(f"{pred[i]}" + '\n')