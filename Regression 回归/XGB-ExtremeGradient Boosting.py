# XGBoost 回归
from re import X
from xgboost import XGBRegressor
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import cross_val_score, cross_val_predict, KFold
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score as R2
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split

# 加载本地未划分的数据集
X =  pd.read_csv('california_housing_X.csv')
y_DataFrame =  pd.read_csv('california_housing_y.csv')
# 将y的特征名称，如“Price”写入到这里
y = y_DataFrame['Price']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#定义XGB回归对象,进行实例化
xgbr = XGBRegressor(learning_rate= 0.09924224,
            #max_depth=3,
            #min_child_weight=0.8251295,
            n_estimators=190,
            #subsample= 0.21959604026,
            #gamma=0.0049220938754,
            reg_alpha = 0.002834496,
            reg_lambda = 0.00018473,
            early_stopping_rounds=None,
            objective='reg:squarederror',
            booster='gblinear')   #booster指定弱学习器的类型，默认值为 ‘ gbtree ’，表示使用基于树的模型进行计算。还可以选择为 ‘gblinear’ 表示使用线性模型作为弱学习器。
#print(xgbr)

xgbr.fit(X_train, y_train) #校正建模，训练
y_predict_train = xgbr.predict(X_train)  #根据X_train得到的预测结果
y_predict_test = xgbr.predict(X_test)  #根据X_test得到的预测结果 (传统接口predict)

# 建模/校正集_train评估
rmsep = np.sqrt(MSE(y_train, y_predict_train))  #预测集RMSEP
rp2 = R2(y_train, y_predict_train) #R2(决定系数),sklearn.metrics.r2_score使用的是（参考值，预测值）
print("RMSE_C: %.4f" % rmsep, "~", "Rcal^2: %.4f" % rp2)

# K折-交叉验证-K-fold
# kfold = KFold(n_splits=10, shuffle=True) #cv=kfold注释掉，不让他们“shuffle洗牌”, 以保证结果一致
mse_scores = cross_val_score(xgbr, X_train, y_train, cv=10, scoring='neg_mean_squared_error')  #scoring参数 https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter 
rmsep = np.sqrt(-mse_scores)
kf_cv_scores = cross_val_score(xgbr, X_train, y_train, cv=10, scoring='r2')  #cross_val_score的默认scoring就是r2
print("RMSE_CV: %.4f" % rmsep.mean(), "~", "Rcv^2: %.4f" % kf_cv_scores.mean())

# 预测集_test评估
rmsep = np.sqrt(MSE(y_test, y_predict_test))  #预测集RMSEP
rp2 = R2(y_test, y_predict_test) #R2(决定系数),sklearn.metrics.r2_score参数使用的是（参考值，预测值）
print("RMSE_P: %.4f" % rmsep,"~", "Rpre^2: %.4f" % rp2)


# 实际值 vs 预测值散点图
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_predict_test, color="green", s=60, edgecolor="black")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2)
plt.title("Actual vs Predicted Values (XGBoost)", fontsize=16)
plt.xlabel("Actual Values", fontsize=14)
plt.ylabel("Predicted Values", fontsize=14)
plt.grid(True)
plt.show()

# 残差图
plt.figure(figsize=(10, 6))
sns.residplot(x=y_test, y=y_predict_test, lowess=True, color="orange", line_kws={'color': 'red', 'lw': 2})
plt.title("Residual Plot (XGBoost)", fontsize=16)
plt.xlabel("Actual Values", fontsize=14)
plt.ylabel("Residuals", fontsize=14)
plt.grid(True)
plt.show()

