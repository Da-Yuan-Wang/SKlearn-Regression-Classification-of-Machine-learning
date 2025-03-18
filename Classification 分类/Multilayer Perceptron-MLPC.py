# 导入所需的库
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# 加载本地未划分的数据集
X =  pd.read_csv('iris_X.csv')
y =  pd.read_csv('iris_y.csv')
# 获取csv文件中的首行特征变量, 将csv中首行的str类型内容存储到list变量中
with open('iris_X.csv', newline='', encoding='utf-8') as csvfile:
    first_row = csv.reader(csvfile)  # 读取第一行
    feature_names = next(first_row)  
# 手动写上分类目标的类别名称，按文件中 0,1,2,…… 的类别顺序
target_names = ['setosa', 'versicolor', 'virginica']

# 划分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 创建并训练MLP神经网络模型
mlp_model = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000, random_state=42)
mlp_model.fit(X_train, y_train)

# 预测测试数据
y_pred = mlp_model.predict(X_test)

# 计算准确度
accuracy = accuracy_score(y_test, y_pred)

# 生成混淆矩阵
conf_matrix = confusion_matrix(y_test, y_pred)

# 打印分类报告
class_report = classification_report(y_test, y_pred, target_names=target_names)

# 绘制图表
plt.figure(figsize=(12, 6))

# 绘制混淆矩阵
plt.subplot(121)
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix of Multilayer Perceptron')
plt.colorbar()
tick_marks = np.arange(len(target_names))
plt.xticks(tick_marks, target_names, rotation=45)
plt.yticks(tick_marks, target_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

for i in range(len(target_names)):
    for j in range(len(target_names)):
        plt.text(j, i, format(conf_matrix[i, j], 'd'), horizontalalignment="center", color="white" if conf_matrix[i, j] > conf_matrix.max() / 2 else "black")

# 绘制分类报告
plt.subplot(122)
x = range(len(target_names))
# 这是针对3分类绘制柱状图，如果多余3个类别，在plt.bar()里面继续增加, float(class_report.split()[20]), float(class_report.split()[25]),…… 依次类推
plt.bar(x, [float(class_report.split()[5]), float(class_report.split()[10]), float(class_report.split()[15])])
plt.xticks(x, target_names)
plt.xlabel('Class')
plt.ylabel('F1-Score')
plt.title('F1-Score by Class')

# 打印模型评估结果
print(f"Accuracy: {accuracy:.2f}")
print(f"Confusion Matrix:\n{conf_matrix}")
print(f"Classification Report:\n{class_report}")

plt.tight_layout()
plt.show()

