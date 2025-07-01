# 14:38 2025/6/30：tested by wky
# 决策树分类器示例：Iris 鸢尾花数据集
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import pandas as pd

# 1. 加载鸢尾花数据集
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target)

# 2. 划分训练集和测试集（80% 训练，20% 测试）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 创建决策树分类器并训练
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# 4. 对测试集进行预测
y_pred = clf.predict(X_test)

# 5. 评估模型准确率
print("模型准确率:", accuracy_score(y_test, y_pred))
print("\n分类报告:\n", classification_report(y_test, y_pred, target_names=iris.target_names))

# 6. 可视化决策树结构（非常直观！）
plt.figure(figsize=(12, 8))
plot_tree(clf, feature_names=iris.feature_names, class_names=iris.target_names,
          filled=True, rounded=True)
plt.title("决策树可视化")
plt.show()
