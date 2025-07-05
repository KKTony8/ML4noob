# 22:34 2025/6/30 tested by wky
### sklearn pipeline是怎么回事？相当于是利用元组简化了对象的创建
    # scaler = StandardScaler()
    # X_train_scaled = scaler.fit_transform(X_train)
    # X_test_scaled = scaler.transform(X_test)

    # model = SVC()
    # model.fit(X_train_scaled, y_train)
    # y_pred = model.predict(X_test_scaled)

    # pipeline = Pipeline([
        # ('scaler', StandardScaler()),
        # ('svc', SVC())
    # ])
    # pipeline.fit(X_train, y_train)
    # y_pred = pipeline.predict(X_test)

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#————————————————————————————————————————————————————————————————————————
# 第一步： 模拟数据：房屋面积 (平方米)、房间数、楼层、建造年份、位置（类别变量）
#————————————————————————————————————————————————————————————————————————
data = {
    'area': [70, 85, 100, 120, 60, 150, 200, 80, 95, 110],
    'rooms': [2, 3, 3, 4, 2, 5, 6, 3, 3, 4],
    'floor': [5, 2, 8, 10, 3, 15, 18, 7, 9, 11],
    'year_built': [2005, 2010, 2012, 2015, 2000, 2018, 2020, 2008, 2011, 2016],
    'location': ['Chaoyang', 'Haidian', 'Chaoyang', 'Dongcheng', 'Fengtai', 'Haidian', 'Chaoyang', 'Fengtai', 'Dongcheng', 'Haidian'],
    'price': [5000000, 6000000, 6500000, 7000000, 4500000, 10000000, 12000000, 5500000, 6200000, 7500000]  # 房价（目标变量）
}

# 创建 DataFrame
df = pd.DataFrame(data)

# 查看数据
print("数据预览：")
print(df.head())


#————————————————————————————————————————————————————————————————————————
# 2. 数据预处理，特征选择
#————————————————————————————————————————————————————————————————————————
X = df[['area', 'rooms', 'floor', 'year_built', 'location']]  # 特征
y = df['price']  # 目标变量

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 构建预处理步骤
numeric_features = ['area', 'rooms', 'floor', 'year_built']
categorical_features = ['location']

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())  # 数值特征标准化
])

# 对多分类数据进行one-hot编码
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))  # 处理测试集中的新类别
])

# 组合成 ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

### 展示一下转换后
X_train_transformed = preprocessor.fit_transform(X_train)
# 查看数据预处理后的结构： 因为location有四个【0，0，1，0】故最后的数据有8列。
print("预处理后的训练数据：")
print(X_train_transformed)

#————————————————————————————————————————————————————————————————————————
# 3. 建立模型
#————————————————————————————————————————————————————————————————————————
# 构建一个包含预处理和回归模型的 Pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),  # 数据预处理步骤
    ('regressor', LinearRegression())  # 回归模型
])

# 训练模型
model_pipeline.fit(X_train, y_train)

# 进行预测
y_pred = model_pipeline.predict(X_test)

# 输出预测结果
print("\n预测结果：")
print(y_pred)


#————————————————————————————————————————————————————————————————————————
# 4. 评估模型
#————————————————————————————————————————————————————————————————————————
# 计算均方误差（MSE）和决定系数（R²）
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# 输出评估结果
print("\n模型评估：")
print(f"均方误差 (MSE): {mse:.2f}")
print(f"决定系数 (R²): {r2:.2f}")
#————————————————————————————————————————————————————————————————————————
# 5. 模型优化：使用网格搜索调整超参数
#————————————————————————————————————————————————————————————————————————
# 对线性回归的超参数进行调优（仅调整 'fit_intercept'）
param_grid = {
    'regressor__fit_intercept': [True, False],  # 是否拟合截距
}

grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=1)
grid_search.fit(X_train, y_train)

# 输出最佳参数和结果
print("\n最佳参数：")
print(grid_search.best_params_)

# 使用最佳模型进行预测
best_model = grid_search.best_estimator_
y_pred_optimized = best_model.predict(X_test)

# 输出优化后的评估结果
mse_opt = mean_squared_error(y_test, y_pred_optimized)
r2_opt = r2_score(y_test, y_pred_optimized)

print("\n优化后的模型评估：")
print(f"均方误差 (MSE): {mse_opt:.2f}")
print(f"决定系数 (R²): {r2_opt:.2f}")
