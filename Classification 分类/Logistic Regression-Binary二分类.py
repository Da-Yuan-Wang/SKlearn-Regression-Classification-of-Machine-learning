'逻辑回归是一个用于二分类(binary classification)的算法。'
'逻辑回归(Logistics Regression), 逻辑回归虽然叫回归, 但实际上属于分类算法, 常用于二分类的任务。'
'当然逻辑回归也可以用于多分类, 这就需要加上其它的方法。'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score
from sklearn.datasets import load_breast_cancer

# 加载乳腺癌数据集
cancer = load_breast_cancer()
X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
y = pd.Series(cancer.target, name='CancerType')

# 数据标准化
X = (X - X.mean()) / X.std()

# 数据可视化：相关性矩阵
plt.figure(figsize=(12, 10))
sns.heatmap(X.corr(), annot=False, cmap='coolwarm')
plt.title("Feature Correlation Matrix", fontsize=16)
plt.show()

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建逻辑回归模型
log_reg = LogisticRegression(max_iter=10000)
log_reg.fit(X_train, y_train)

# 进行预测
y_pred = log_reg.predict(X_test)
y_pred_prob = log_reg.predict_proba(X_test)[:, 1]

# 混淆矩阵
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix", fontsize=16)
plt.xlabel("Predicted Label", fontsize=14)
plt.ylabel("True Label", fontsize=14)
plt.show()

# ROC曲线
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = roc_auc_score(y_test, y_pred_prob)

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.title("ROC Curve", fontsize=16)
plt.xlabel("False Positive Rate", fontsize=14)
plt.ylabel("True Positive Rate", fontsize=14)
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# 特征系数
plt.figure(figsize=(10, 6))
coefficients = pd.Series(log_reg.coef_[0], index=X.columns)
coefficients.plot(kind='bar', color='magenta')
plt.title("Logistic Regression Coefficients", fontsize=16)
plt.xlabel("Features", fontsize=14)
plt.ylabel("Coefficient Value", fontsize=14)
plt.grid(True)
plt.show()

# 模型性能报告
print(classification_report(y_test, y_pred))