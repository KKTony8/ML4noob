# 第0步： 创建必要环境: 进入cmd，创建虚拟环境venv，然后pip install scikit-learn
from sklearn.datasets import load_iris
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 第一步：加载数据，加载鸢尾花数据集
iris = load_iris()

# 将数据转化为 pandas DataFrame
X = pd.DataFrame(iris.data, columns=iris.feature_names)  # 特征数据
y = pd.Series(iris.target)  # 标签数据

###############################穿插一段什么是Series数据?即带标签的一维数组
#ages = pd.Series([25, 30, 22, 40], index=['小明', '小红', '小强', '小李'])
#print(ages)
#print(ages['小明'])

# 显示前五行数据
print(X.head())
print(y.value_counts())
# 划分训练集和测试集（80% 训练集数据和标签，20% 测试集数据和标签）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 第二步： 标准化特征，训练数据fit到的参数，再transform直接去标准化测试数据
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 第三步：确定初始模型：创建 KNN 分类器
knn = KNeighborsClassifier(n_neighbors=3)

# 第四步：我想看看训练多久导入时间模块，训练模型得到最终模型
import time

start = time.time()
knn.fit(X_train, y_train)
end = time.time()
print(f"Training time: {end - start:.4f} seconds")

# 第五步: 测试集评估
y_pred = knn.predict(X_test)

# 类别数量差不多直接，计算准确率评估，如果不平衡或者多分类，可以引入F1分数，recall和precision评估
accuracy = accuracy_score(y_test, y_pred)
print(f'模型准确率: {accuracy:.2f}')

#######################################可视化
# 可视化 - 这里只是绘制了前两个特征，总共有四个特征
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='viridis', marker='o')
plt.title("KNN Classification Results")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()