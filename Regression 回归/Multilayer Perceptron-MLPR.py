# BPNN-MLPRegressor
# SKlearn的MLP建立的模型就是是BP神经网络
from sklearn.neural_network import MLPRegressor
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, cross_val_predict, KFold
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score as R2
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

'''
# 加载加利福尼亚房价数据集
housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = pd.Series(housing.target, name='Price')


# 加载已划分的数据集
X_train =np.loadtxt(open('NIR-SPA-8_Xcal-112.csv',"rb"),delimiter=",",skiprows=1) #CSV文件转化为数组
X_test =np.loadtxt(open('NIR-SPA-8_Xval-60.csv',"rb"),delimiter=",",skiprows=1) #CSV文件转化为数组
y_train =np.loadtxt(open('NIR-Ycal-112.csv',"rb"),delimiter=",",skiprows=1) #CSV文件转化为数组
y_test =np.loadtxt(open('NIR-Yval-60.csv',"rb"),delimiter=",",skiprows=1) #CSV文件转化为数组
'''

# 加载本地未划分的数据集
X =  pd.read_csv('california_housing_X.csv')
y_DataFrame =  pd.read_csv('california_housing_y.csv')
# 将y的特征名称，如“Price”写入到这里
y = y_DataFrame['Price']


# 数据标准化
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)


# 数据可视化：相关性矩阵
plt.figure(figsize=(10, 8))
sns.heatmap(X.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Matrix", fontsize=16)
plt.show()

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

print('X_train样本:',X_train.shape,'X_test样本:',X_test.shape)

bpnnr =  MLPRegressor(hidden_layer_sizes=(10,10), 
                 activation='relu',    #隐藏层的激活函数
                 solver='lbfgs',       #对于小型数据集，“lbfgs”可以更快地收敛并表现更好
                 alpha=0.01,           #L2惩罚（正则项）参数。
                 batch_size='auto',
                 learning_rate='constant',  #' constant '是一个恒定的学习速率，由' learning_rate_init '给出
                 learning_rate_init=0.01,  #使用的初始学习率。它控制更新权重的步长。仅在Solver ='sgd'或'adam'时使用。
                 power_t=0.5, 
                 max_iter=10000,          #最大迭代次数
                 shuffle=False, 
                 # random_state=RandomState,  #确定权重和偏差初始化的随机数生成，如果使用提前停止，则确定训练测试拆分，以及求解器 ='sgd' 或'adam' 时的批量采样。
                 tol=0.0001, 
                 verbose=False, 
                 warm_start=False, 
                 momentum=0.9, 
                 nesterovs_momentum=True, 
                 early_stopping=False, 
                 validation_fraction=0.1, 
                 beta_1=0.9, 
                 beta_2=0.999, 
                 epsilon=1e-08, 
                 n_iter_no_change=10, 
                 max_fun=15000)  #定义MLPR回归对象,进行实例化
#进行标准化，数据预处理
regr = make_pipeline(StandardScaler(), bpnnr)  #查看https://scikit-learn.org/stable/modules/svm.html#tips-on-practical-use
regr.fit(X_train, y_train) #校正建模，训练,返回训练好的模型
y_predict_train = regr.predict(X_train)  #根据X_train得到的预测结果
y_predict_test = regr.predict(X_test)  #根据X_test得到的预测结果 (传统接口predict)

# 建模/校正集_train评估
rmsec = np.sqrt(MSE(y_train, y_predict_train))  #预测集RMSEP
rc2 = regr.score(X_train, y_train) #Xarray-like of shape (n_samples, n_features);yarray-like of shape (n_samples,) or (n_samples, n_outputs)True values for X.（X值，y真值/参考值）

# K折-交叉验证-K-fold
# K折-交叉验证-K-fold,多种方式：model_selection.cross_val_score 和 model_selection.cross_validate
# kfold = KFold(n_splits=10, shuffle=True) #cv=kfold注释掉, 不让他们“shuffle洗牌”, 以保证结果一致
cv_scores = cross_validate(regr, X_train, y_train, cv=10,  #【cross_validate】https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html#sklearn.model_selection.cross_validate
                        scoring=('r2', 'neg_mean_squared_error'),
                        return_train_score=True,
                        n_jobs = -1
                        )  #scoring参数 https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter 
#sorted(cv_scores.keys())    #在此后一行运行到此，查看cv_scores的返回值——字典dict值：https://scikit-learn.org/stable/modules/cross_validation.html#the-cross-validate-function-and-multiple-metric-evaluation

rmse_cv =  np.sqrt(-(cv_scores['test_neg_mean_squared_error']).mean())
rcv_2 = (cv_scores['test_r2']).mean()
# 预测集_test评估
rmsep = np.sqrt(MSE(y_test, y_predict_test))  #预测集RMSEP
rp2 = regr.score(X_test, y_test) #R2(决定系数)


print("RMSE_C: %.2f" % rmsec, "--", "Rc^2: %.2f" % rc2)
print("RMSE_CV: %.2f" % rmse_cv, "--", "Rcv^2: %.2f" % rcv_2)
print("RMSE_P: %.2f" % rmsep,"--", "Rp^2: %.2f" % rp2)
# print("BPNN的参数:", regr.get_params())
print("………………………………………………………………………………")


# 实际值 vs 预测值散点图
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_predict_test, color="blue", s=60, edgecolor="black")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2)
plt.title("Actual vs Predicted Values", fontsize=16)
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

