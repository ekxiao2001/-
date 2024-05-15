import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

def classification_without_clustering(rf_classifier, X_train, X_test, y_train, y_test):
    # 训练随机森林分类器
    rf_classifier.fit(X_train, y_train)
    # 在测试集上进行预测
    y_pred = rf_classifier.predict(X_test)
    # 计算分类准确率
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

def classification_with_clustering(rf_classifier, X_train, X_test, y_train, y_test):
    #TODO
    # 聚类 + 添加新的额外特征
    kmeans = KMeans(n_clusters=8, random_state=20).fit(X_train) # 构建kmeans
    X_train_with_cluster = X_train.copy()
    X_train_with_cluster['cluster'] = kmeans.labels_    # 将聚类的标签值作为新的特征放入训练数据中

    X_test_with_cluster = X_test.copy()
    X_test_with_cluster['cluster'] = kmeans.predict(X_test) # 将聚类的标签值作为新的特征放入测试数据中

    # 训练随机森林分类器
    rf_classifier.fit(X_train_with_cluster, y_train)
    # 在测试集上进行预测
    y_pred = rf_classifier.predict(X_test_with_cluster)
    # 计算分类准确率
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy


def main():
    # 读取训练集和测试集
    train_data = pd.read_csv('user_train.csv')
    test_data = pd.read_csv('user_test.csv')
    
    # 分割特征和目标变量
    X_train = train_data.drop('Label', axis=1)
    y_train = train_data['Label']
    X_test = test_data.drop('Label', axis=1)
    y_test = test_data['Label']
    
    # 创建一个随机森林分类器对象
    rf_classifier = RandomForestClassifier(random_state=20)
    accuracy1 = classification_without_clustering(rf_classifier, X_train, X_test, y_train, y_test)
    accuracy2 = classification_with_clustering(rf_classifier, X_train, X_test, y_train, y_test)
    print("聚类前分类准确率：", accuracy1)
    print("聚类后分类准确率：", accuracy2)

if __name__ == '__main__':
    main()