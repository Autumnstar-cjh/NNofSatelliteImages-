import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np

# 读取自变量数据表
independent_variables = pd.read_csv('FL_DataSet/FL_allmatch_data_train.csv')

# Reading the dependent variable
target = pd.read_csv('FL_Dataset/FL_allmatch_data_train.csv')['travel_driving_ratio']

data = independent_variables[['edu_bachelor_ratio','edu_master_ratio','edu_phd_ratio','employment_unemployed_ratio','household_size_avg','inc_median_ind','race_black_ratio','race_white_ratio','sex_male_ratio','vehicle_per_capita']
]

# Loading the new variables
new_variables = pd.read_csv('FL_DataSet/train_class_ratios.csv').drop(columns=['Image Name'])

# Concatenating the new variables with the existing DataFrame
# Make sure that both data and new_variables have the same index or are aligned correctly
data_with_new_vars = pd.concat([data, new_variables], axis=1)

# 分割数据为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(data_with_new_vars, target, test_size=0.2, random_state=42)

# 创建Lasso回归模型实例
lasso = Lasso(alpha=0.0001)  # alpha值作为正则化强度参数

# 训练模型
lasso.fit(X_train, y_train)

# 使用模型对测试集进行预测
y_pred = lasso.predict(X_test)

# 计算关键指标
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

# 打印模型摘要
print("Lasso Regression Model Summary:")
print("Coefficients:", lasso.coef_)
print("Intercept:", lasso.intercept_)
print("R^2 Score:", r2)
print("Mean Squared Error:", mse)

# 如果需要，也可以将这些结果写入文件
with open('Lasso_Model_Summary.txt', 'w') as f:
    f.write("Lasso Regression Model Summary:\n")
    f.write("Coefficients: {}\n".format(lasso.coef_))
    f.write("Intercept: {:.5f}\n".format(lasso.intercept_))
    f.write("R^2 Score: {:.5f}\n".format(r2))
    f.write("Mean Squared Error: {:.5f}\n".format(mse))

# 计算样本大小 n 和特征数量 p
n = X_train.shape[0]  # 样本大小
p = X_train.shape[1]  # 特征数量（自变量的数量）

# 计算调整后的R方
adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))

# 打印调整后的R方
print("Adjusted R^2 Score:", adjusted_r2)

# 将调整后的R方写入文件
with open('Lasso_Model_Summary.txt', 'a') as f:  # 注意这里使用'a'模式追加内容
    f.write("Adjusted R^2 Score: {:.5f}\n".format(adjusted_r2))