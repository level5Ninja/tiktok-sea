from scipy import stats
import pandas as pd
df = pd.read_excel("/Users/antonio/Desktop/TIKTOK project/AB_test_data.xlsx")

# 假设 active_day 为关键指标
exp_active = df[df['exp_group']=='exp_group']['active_day']
ctrl_active = df[df['exp_group']=='control_group']['active_day']

print("数据维度: ", exp_active.shape)
print("数据维度: ", ctrl_active.shape)


# 计算均值和标准差
mean_exp, std_exp = exp_active.mean(), exp_active.std()
mean_ctrl, std_ctrl = ctrl_active.mean(), ctrl_active.std()

std_exp = exp_active.std()
std_ctrl = ctrl_active.std()

print("实验组7日活跃天数标准差:", std_exp)
print("对照组7日活跃天数标准差:", std_ctrl)

# 独立样本t检验
t_stat, p_value = stats.ttest_ind(exp_active, ctrl_active)
print(f"实验组均值: {mean_exp:.2f}, 对照组均值: {mean_ctrl:.2f}")
print(f"t统计量: {t_stat:.2f}, p值: {p_value:.4f}")

