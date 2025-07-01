# 14:38 2025/6/30：tested by wky
# Kmeans的演示
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 1. 加载数据，无需pandas的dataframe
iris = load_iris()
X = iris.data
y_true = iris.target  # 真实标签（只用于对比，不用于训练）
print(X[:5]) 

# 2. 使用 KMeans 聚类
kmeans = KMeans(n_clusters=3, random_state=42)
y_kmeans = kmeans.fit_predict(X)  # 聚类结果标签

# 3. 用 PCA 将4维数据降到2维，便于可视化对比，放在Kmeans训练之后
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# 4. 画图对比：左图是 KMeans 聚类结果，右图是真实分类
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# 左图：聚类结果
ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=y_kmeans, cmap='viridis', s=50)
ax1.set_title("KMeans")
ax1.set_xlabel("PCA Component 1")
ax1.set_ylabel("PCA Component 2")

# 右图：真实标签
ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=y_true, cmap='Set1', s=50)
ax2.set_title("test")
ax2.set_xlabel("PCA Component 1")
ax2.set_ylabel("PCA Component 2")

plt.tight_layout()
plt.show()

