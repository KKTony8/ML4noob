from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. 加载数据
iris = load_iris()
X = iris.data
y = iris.target

# 2. 划分训练集和测试集，30%测试
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. 定义SVM模型，使用线性核
model = SVC(kernel='linear')

# 4. 训练模型
model.fit(X_train, y_train)

# 5. 预测测试集
y_pred = model.predict(X_test)

# 6. 计算准确率
acc = accuracy_score(y_test, y_pred)
print(f"测试集准确率: {acc:.2f}")

# 7. 输出前5个预测结果和真实标签对比
print("预测结果:", y_pred[:5])
print("真实标签:", y_test[:5])
