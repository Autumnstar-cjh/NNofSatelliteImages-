import pandas as pd
import statsmodels.api as sm

# 读取自变量数据表
independent_variables = pd.read_csv('FL_DataSet/FL_allmatch_data_train.csv')

# Reading the dependent variable
target = pd.read_csv('FL_Dataset/FL_allmatch_data_train.csv')['travel_driving_ratio']

# 选择自变量
data = independent_variables[['edu_bachelor_ratio', 'edu_master_ratio', 'edu_phd_ratio',
                              'employment_unemployed_ratio', 'household_size_avg', 'inc_median_ind',
                              'race_black_ratio', 'race_white_ratio', 'sex_male_ratio', 'vehicle_per_capita']]

# Loading the new variables
new_variables = pd.read_csv('FL_DataSet/train_class_ratios.csv').drop(columns=['Image Name'])

# Concatenating the new variables with the existing DataFrame
data_with_new_vars = pd.concat([data, new_variables], axis=1)

# 添加常数项，以便拟合截距项
data_with_new_vars = sm.add_constant(data_with_new_vars)

# 创建线性回归模型
model = sm.OLS(target, data_with_new_vars)

# 拟合模型
results = model.fit()

# 输出模型摘要
print(results.summary())

# 如果需要，也可以将模型摘要写入文件
with open('OLS_Baseline_Summary.txt', 'w') as f:
    f.write(results.summary().as_text())

