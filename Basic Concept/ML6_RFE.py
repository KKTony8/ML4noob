# 数据预处理中特征选择的方法 RFE： (recursive feature elimination) 递归特征消除
# 可以先采用一个模型作为基模型，然后用RFE方法去剔除掉权重较小的特征
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE

# 加载示例数据集（鸢尾花）
data = load_iris()
X = data.data
y = data.target

# 用随机森林做基模型
clf = RandomForestClassifier(random_state=42)

# RFE，选3个最重要特征
rfe = RFE(estimator=clf, n_features_to_select=3)
X_rfe = rfe.fit_transform(X, y)

print("被选择的特征索引:", rfe.get_support(indices=True))
print("原特征数:", X.shape[1])
print("选择后特征数:", X_rfe.shape[1])

