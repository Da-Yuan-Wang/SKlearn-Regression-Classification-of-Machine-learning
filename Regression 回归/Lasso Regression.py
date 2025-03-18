import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score, cross_val_predict, KFold
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score as R2
from sklearn.datasets import load_diabetes

# 加载糖尿病数据集
diabetes = load_diabetes()
# X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
'''
# 添加噪声特征
np.random.seed(42)
noise = np.random.randn(X.shape[0], 10)
X_noise = pd.DataFrame(noise, columns=[f'Noise_{i}' for i in range(10)])
X = pd.concat([X, X_noise], axis=1)
'''
# y = pd.Series(diabetes.target, name='DiseaseProgression')

# 加载本地未划分的数据集
X =  pd.read_csv('california_housing_X.csv')
y_DataFrame =  pd.read_csv('california_housing_y.csv')
# 将目标变量Price添加到数据集中（将y的特征名称，如“Price”写入到这里）
y = y_DataFrame['Price']

# 数据标准化
X = (X - X.mean()) / X.std()

# 数据可视化：相关性矩阵
plt.figure(figsize=(12, 10))
sns.heatmap(X.corr(), annot=False, cmap='coolwarm')
plt.title("Feature Correlation Matrix", fontsize=16)
plt.show()

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建套索回归模型
model = Lasso(alpha=0.1)
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


# 实际值 vs 预测值散点图
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_predict_test, color="blue", s=60, edgecolor="black")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2)
plt.title("Actual vs Predicted Values (Lasso Regression)", fontsize=16)
plt.xlabel("Actual Values", fontsize=14)
plt.ylabel("Predicted Values", fontsize=14)
plt.grid(True)
plt.show()

# 残差图
plt.figure(figsize=(10, 6))
sns.residplot(x=y_test, y=y_predict_test, lowess=True, color="purple", line_kws={'color': 'red', 'lw': 2})
plt.title("Residual Plot (Lasso Regression)", fontsize=16)
plt.xlabel("Actual Values", fontsize=14)
plt.ylabel("Residuals", fontsize=14)
plt.grid(True)
plt.show()

# 套索回归系数
plt.figure(figsize=(10, 6))
coefficients = pd.Series(model.coef_, index=X.columns)
coefficients.plot(kind='bar', color='coral')
plt.title("Lasso Regression Coefficients", fontsize=16)
plt.xlabel("Features", fontsize=14)
plt.ylabel("Coefficient Value", fontsize=14)
plt.grid(True)
plt.show()

