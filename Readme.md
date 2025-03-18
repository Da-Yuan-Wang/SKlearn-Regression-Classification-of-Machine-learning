# Regression and classification based on Scikit-learn library using training with local csv data
==============================================================
# 基于Scikit-learn库的回归和分类，使用本地csv数据训练
https://github.com/Da-Yuan-Wang/SKlearn-Regression-Classification-of-Machine-learning

%【联系作者】：王大元，江南大学，15553268586，E-mail: wang3da2_yuan1@163.com

%【Contact Author】：Dayuan Wang, State Key Laboratory of Food Science and Resources, School of Food Science and Technology, Jiangnan University, 214122 Wuxi, Jiangsu, China [E-mail]:wang3da2_yuan1@163.com 
______________________________
安装VScode或Pycharm社区版，安装必要插件：python、chinese、vscode-icons、rainbow CSV、intellicode、github copilot等

安装Anaconda及环境配置：https://blog.csdn.net/aiyouwei_66/article/details/135115416

Sklearn网站：https://scikit-learn.org/

Sklearn监督学习：https://scikit-learn.org/stable/supervised_learning.html

______________________________
配置Anaconda的电脑环境变量(根据自己实际安装Anaconda路径，写入环境变量的Path)：

D:\Anaconda

D:\Anaconda\Scripts

D:\Anaconda\Library\mingw-w64\bin

D:\Anaconda\Library\bin

______________________________
conda env list

conda create --name sklearn python=3.10

conda activate sklearn

pip list

pip install scikit-learn==1.6.1

pip install matplotlib==3.10.1

pip install statsmodels==0.14.4

pip install seaborn==0.13.2

pip show 库名

______________________________
运行时，如需中断，在终端或命令行界面（多次）按下"Ctrl+C"组合键，出现KeyboardInterrupt：异常，停止执行。
To interrupt the execution, press Ctrl+C multiple times on the terminal or command-line interface (CLI). An exception occurs with KeyboardInterrupt, stopping the execution.

==============================================

【分类Classfication】

Iris数据集是常用的分类实验数据集，由Ronald Fisher在1936年收集整理。Iris也称鸢尾花卉数据集，是一类多重变量分析的数据集。
Iris数据集包含150个数据样本，分为3类，每类50个数据，每个数据包含4个属性。
可通过花萼长度，花萼宽度，花瓣长度，花瓣宽度4个属性预测鸢尾花卉属于（Setosa，Versicolour，Virginica）三个种类中的哪一类。
______________________________

红酒口感数据集包括将近 1599 种红酒的数据。每一种红酒都有一系列化学成分的测量指标，包括酒精含量、挥发性酸、亚硝酸盐。每种红酒都有一个口感评分值，是三个专业评酒员的评分的平均值。
每种样本都由专家做了质量评级，并进行了理化指标检验。包含以下12个字段：
fixed acidity 固定酸度
volatile acidity 挥发性酸度
citric acid 柠檬酸
residual sugar 残糖
chlorides 氯化物
free sulfur dioxide 游离二氧化硫
total sulfur dioxide 总二氧化硫
density 密度
pH pH值
sulphates 硫酸盐
alcohol 酒精度
quality 质量 - 0 到 10 之间的得分（葡萄酒专家至少 3 次评估的中值）



【回归Regression】

‌‌California Housing Prices dataset‌是加利福尼亚房价数据集‌，这个数据集包含了一些关于加州各小区的数据，如人口、房屋面积、犯罪率等，这些都是影响房价的重要因素。该数据集由美国统计局提供，通常被用于探索和预测房价‌。加利福尼亚房价信息数据集（fetch_california_housing） 是一个备受推崇的开源数据集，广泛应用于教学和演示。该数据集包含了加利福尼亚州不同区域的住房价格信息，为回归分析和模型训练提供了丰富的数据支持。
加利福尼亚房价信息数据集包含了以下关键字段：
经度（Longitude） 和 纬度（Latitude）：用于定位房屋的地理位置。
房龄（Housing Median Age）：房屋的中位年龄，反映了房屋的新旧程度。
房间总数（Total Rooms） 和 卧室总数（Total Bedrooms）：提供了房屋的规模信息。
人口（Population） 和 家庭总数（Households）：反映了社区的人口密度和家庭结构。
收入中位数（Median Income）：衡量了该地区的经济水平。
房价中位数（Median House Value）：作为目标变量，用于预测和分析。
______________________________

diabetes 是一个关于糖尿病的数据集， 该数据集包括442个病人的生理数据及一年以后的病情发展情况。
该数据集共442条信息，特征值总共10项, 如下:
age: 年龄
sex: 性别
bmi(body mass index): 身体质量指数
bp(blood pressure): 血压（平均血压）
s1——tc，T细胞（一种白细胞）
s2——ldl，低密度脂蛋白
s3——hdl，高密度脂蛋白
s4——tch，促甲状腺激素
s5——ltg，拉莫三嗪
s6——glu，血糖水平
% s1,s2,s3,s4,s4,s6: 六种血清的化验数据。%
Targets：25-346间整数，表示一年后患疾病的定量指标
______________________________________________
