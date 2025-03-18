import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score, cross_val_predict, KFold
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score as R2
from sklearn.datasets import fetch_california_housing

# 加载加利福尼亚房价数据集
housing = fetch_california_housing()
# X = pd.DataFrame(housing.data, columns=housing.feature_names)
# y = pd.Series(housing.target, name='Price')

# 加载本地未划分的数据集
X =  pd.read_csv('california_housing_X.csv')
y_DataFrame =  pd.read_csv('california_housing_y.csv')
# 将y的特征名称，如“Price”写入到这里
y = y_DataFrame['Price']

# 数据标准化
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建决策树回归模型
model = DecisionTreeRegressor(max_depth=8, random_state=42)
model.fit(X_train, y_train)

# 进行预测
y_predict_train = model.predict(X_train)  #根据X_train得到的预测结果
y_predict_test = model.predict(X_test)  #根据X_test得到的预测结果 (传统接口predict)

# 输出训练集_train评估
rmsep = np.sqrt(MSE(y_train, y_predict_train))  #预测集RMSEP
rp2 = R2(y_train, y_predict_train) #R2(决定系数),sklearn.metrics.r2_score使用的是（参考值，预测值）
print("RMSE_C: %.2f" % rmsep, "--", "Rcal^2: %.2f" % rp2)

# 交叉验证Cross-validation(CV)评估- K折(K-fold)
# kfold = KFold(n_splits=10, shuffle=True) #cv=kfold注释掉，不让他们“shuffle洗牌”, 以保证结果一致
mse_scores = cross_val_score(model, X_train, y_train, cv=10, scoring='neg_mean_squared_error')  #scoring参数 https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter 
rmsep = np.sqrt(-mse_scores)
kf_cv_scores = cross_val_score(model, X_train, y_train, cv=10, scoring='r2')  #cross_val_score的默认scoring就是r2
print("RMSE_CV: %.2f" % rmsep.mean(), "--", "Rcv^2: %.2f" % kf_cv_scores.mean())

# 输出预测集_test评估
rmsep = np.sqrt(MSE(y_test, y_predict_test))  #预测集RMSEP
rp2 = R2(y_test, y_predict_test) #R2(决定系数),sklearn.metrics.r2_score参数使用的是（参考值，预测值）
print("RMSE_P: %.2f" % rmsep,"--", "Rpre^2: %.2f" % rp2)

# 决策树结构
from sklearn.tree import plot_tree

plt.figure(figsize=(15, 8))
plot_tree(model, feature_names=X.columns, filled=True, rounded=True)
plt.title("Decision Tree Structure", fontsize=16)
plt.show()

# 实际值 vs 预测值散点图
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_predict_test, color="red", s=60, edgecolor="black")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='blue', linewidth=2)
plt.title("Actual vs Predicted Values (Decision Tree Regression)", fontsize=16)
plt.xlabel("Actual Values", fontsize=14)
plt.ylabel("Predicted Values", fontsize=14)
plt.grid(True)
plt.show()

# 残差图
plt.figure(figsize=(10, 6))
sns.residplot(x=y_test, y=y_predict_test, lowess=True, color="green", line_kws={'color': 'red', 'lw': 2})
plt.title("Residual Plot", fontsize=16)
plt.xlabel("Actual Values", fontsize=14)
plt.ylabel("Residuals", fontsize=14)
plt.grid(True)
plt.show()

# 特征重要性
plt.figure(figsize=(12, 8))
importances = pd.Series(model.feature_importances_, index=X.columns)
importances.sort_values().plot(kind='barh', color='orange')
plt.title("Feature Importances (Decision Tree Regression)", fontsize=16)
plt.xlabel("Importance Score", fontsize=14)
plt.ylabel("Features", fontsize=14)
plt.grid(True)
plt.show()
