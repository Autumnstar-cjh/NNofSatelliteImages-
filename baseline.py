import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# 读取自变量数据表
edu_bachelor = pd.read_excel('FL_Dataset/testlabel_edu_bachelor_ratio.xlsx')
edu_master = pd.read_excel('FL_Dataset/testlabel_edu_master_ratio.xlsx')
edu_phd = pd.read_excel('FL_Dataset/testlabel_edu_phd_ratio.xlsx')
employment_unemployed = pd.read_excel('FL_Dataset/testlabel_employment_unemployed_ratio.xlsx')
household_size_avg = pd.read_excel('FL_Dataset/testlabel_household_size_avg.xlsx')
inc_median_ind = pd.read_excel('FL_Dataset/testlabel_inc_median_ind.xlsx')
race_black_ratio = pd.read_excel('FL_Dataset/testlabel_race_black_ratio.xlsx')
race_white_ratio = pd.read_excel('FL_Dataset/testlabel_race_white_ratio.xlsx')
sex_male_ratio = pd.read_excel('FL_Dataset/testlabel_sex_male_ratio.xlsx')
vehicle_per_capita = pd.read_excel('FL_Dataset/testlabel_vehicle_per_capita.xlsx')

# 读取因变量数据表
target = pd.read_excel('FL_Dataset/testlabel.xlsx')['target']

# 合并自变量数据表
data = pd.concat([edu_bachelor['target'], edu_master['target'], edu_phd['target'],
                  employment_unemployed['target'], household_size_avg['target'],
                  inc_median_ind['target'], race_black_ratio['target'], race_white_ratio['target'],
                  sex_male_ratio['target'], vehicle_per_capita['target']], axis=1)

new_variables = pd.read_csv('LoveDA/test_class_ratios.csv').drop(columns=['Image Name'])

data = pd.concat([data, new_variables], axis=1)

# 创建线性回归模型
model = LinearRegression()

# 拟合模型
model.fit(data, target)

# 预测因变量
predictions = model.predict(data)

# 计算 R-squared
r2 = r2_score(target, predictions)

# 打印 R-squared
print("R-squared:", r2)
