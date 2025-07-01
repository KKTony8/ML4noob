# 模型优化中网格搜索用于挑选超参数的方法
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier

# 1. 加载数据
iris = load_iris()
X = iris.data
y = iris.target

# 2. 划分训练集和测试集（80%训练，20%测试）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 定义参数搜索空间
param_grid = {
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 4, 6]
}

# 4. 创建模型和网格搜索对象
clf = DecisionTreeClassifier()
grid_search = GridSearchCV(clf, param_grid, cv=5)

# 5. 进行训练 + 搜索最优超参数
grid_search.fit(X_train, y_train)

# 6. 输出结果
print("最优参数:", grid_search.best_params_)
print("最优得分: %.2f" % grid_search.best_score_)

