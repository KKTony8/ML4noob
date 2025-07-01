# 15:01 2025/6/30：tested by wky
# 单纯用来验证DT模型的准确性
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

# 加载数据
iris = load_iris()
X, y = iris.data, iris.target

# 定义模型
clf = DecisionTreeClassifier()

# 5折交叉验证
scores = cross_val_score(clf, X, y, cv=5)

print("每折得分:", scores)
print("平均得分:", scores.mean())
