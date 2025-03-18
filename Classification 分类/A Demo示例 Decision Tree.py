# 导入所需的库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import plot_tree

''''''
# 加载鸢尾花数据集
iris = datasets.load_iris()
X = iris.data
y = iris.target

print('iris.feature_names是:', iris.feature_names)
print('iris.target_names是:', iris.target_names)

#将X和y数据写入csv文件,并导出到根目录下，用户可查看需准备数据文件的样式。
DataFrame_iris_X = pd.DataFrame(X)
DataFrame_iris_X.to_csv("原始_iris_X.csv", index=False, sep=',')
DataFrame_iris_y = pd.DataFrame(y)
DataFrame_iris_y.to_csv("原始_iris_y.csv", index=False, sep=',')


# 划分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 创建并训练决策树模型
decision_tree_model = DecisionTreeClassifier(random_state=42)
decision_tree_model.fit(X_train, y_train)

# 预测测试数据
y_pred = decision_tree_model.predict(X_test)

# 计算准确度
accuracy = accuracy_score(y_test, y_pred)

# 生成混淆矩阵: True label为行, Predicted label为列
conf_matrix = confusion_matrix(y_test, y_pred) 

# 打印分类报告
class_report = classification_report(y_test, y_pred, target_names=iris.target_names)

# 绘制图表
plt.figure(figsize=(12, 6))

# 绘制混淆矩阵
plt.subplot(121)
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix of Decision Tree')
plt.colorbar()
tick_marks = np.arange(len(iris.target_names))
plt.xticks(tick_marks, iris.target_names, rotation=45)
plt.yticks(tick_marks, iris.target_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')



for i in range(len(iris.target_names)):
    for j in range(len(iris.target_names)):
        plt.text(j, i, format(conf_matrix[i, j], 'd'), horizontalalignment="center", color="white" if conf_matrix[i, j] > conf_matrix.max() / 2 else "black")

# 绘制分类报告
plt.subplot(122)
x = range(len(iris.target_names))
# 这是针对3分类绘制柱状图，如果多余3个类别，在plt.bar()里面继续增加, float(class_report.split()[20]), float(class_report.split()[25]),…… 依次类推
plt.bar(x, [float(class_report.split()[5]), float(class_report.split()[10]), float(class_report.split()[15])])
plt.xticks(x, iris.target_names)
plt.xlabel('Class')
plt.ylabel('F1-Score')
plt.title('F1-Score by Class')

# 打印模型评估结果
print(f"Accuracy: {accuracy:.2f}")
print(f"Confusion Matrix:\n{conf_matrix}")
print(f"Classification Report:\n{class_report}")

plt.tight_layout()
plt.show()

# 绘制决策树
plt.figure(figsize=(12, 6))
plot_tree(decision_tree_model, filled=True, feature_names=iris.feature_names, class_names=iris.target_names)
plt.show()