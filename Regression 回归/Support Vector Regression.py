import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, cross_val_predict, KFold
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score as R2
from sklearn.datasets import make_regression
from sklearn.datasets import fetch_california_housing


# 生成虚拟数据集
X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)

# 创建DataFrame
data = pd.DataFrame(X, columns=[f'Feature_{i}' for i in range(X.shape[1])])
data['Heating Load'] = y

X = data.iloc[:, :-1]
y = data.iloc[:, -1]  # 我们预测 Heating Load

'''
# 加载加利福尼亚房价数据集
california = fetch_california_housing()
# X = pd.DataFrame(california.data, columns=california.feature_names)
# y = pd.Series(california.target, name='MedHouseVal')

# 加载本地未划分的数据集
X =  pd.read_csv('california_housing_X.csv')
y_DataFrame =  pd.read_csv('california_housing_y.csv')
# 将y的特征名称，如“Price”写入到这里
y = y_DataFrame['Price']
'''

# 数据标准化
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# 创建支持向量回归模型
model = SVR(kernel='rbf', C=100, epsilon=0.1)
model.fit(X_train, y_train)

# 进行预测
y_predict_train = model.predict(X_train)  #根据X_train得到的预测结果
y_predict_test = model.predict(X_test)  #根据X_test得到的预测结果 (传统接口predict)

# 逆标准化
y_train = scaler_y.inverse_transform(y_train.reshape(-1, 1)).flatten()
y_test = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
y_predict_train = scaler_y.inverse_transform(y_predict_train.reshape(-1, 1)).flatten()
y_predict_test = scaler_y.inverse_transform(y_predict_test.reshape(-1, 1)).flatten()

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


# 实际值 vs 预测值散点图
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_predict_test, color="green", s=60, edgecolor="black")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2)
plt.title("Actual vs Predicted Values (SVR)", fontsize=16)
plt.xlabel("Actual Values", fontsize=14)
plt.ylabel("Predicted Values", fontsize=14)
plt.grid(True)
plt.show()

# 残差图
plt.figure(figsize=(10, 6))
sns.residplot(x=y_test, y=y_predict_test, lowess=True, color="orange", line_kws={'color': 'red', 'lw': 2})
plt.title("Residual Plot (SVR)", fontsize=16)
plt.xlabel("Actual Values", fontsize=14)
plt.ylabel("Residuals", fontsize=14)
plt.grid(True)
plt.show()
