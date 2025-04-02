import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

# 读取数据文件（请确保文件路径和文件名正确）
df = pd.read_excel("/Users/antonio/Desktop/TIKTOK project/AB_test_data.xlsx")

# 筛选低端设备用户
low_device_df = df[df['device_level'] == 'low']

# 筛选年龄在30-45和60+的用户
age_sensitive_df = df[df['age_level'].isin(['30-45', '60+'])]

def analyze_group(subset_df, group_name):
    """
    对给定数据子集进行分析：比较实验组与对照组的 active_day 均值，计算标准差、t检验并作图。
    """
    # 假设实验组和对照组的标签分别为 'exp_group' 和 'control_group'
    exp_data = subset_df[subset_df['exp_group'] == 'exp_group']['active_day']
    ctrl_data = subset_df[subset_df['exp_group'] == 'control_group']['active_day']
    
    # 如果样本数量不足，跳过分析
    if len(exp_data) < 5 or len(ctrl_data) < 5:
        print(f"{group_name} 样本数量不足，无法进行有效统计")
        return

    # 计算均值和标准差
    mean_exp = exp_data.mean()
    std_exp = exp_data.std()
    mean_ctrl = ctrl_data.mean()
    std_ctrl = ctrl_data.std()
    
    # 独立样本 t 检验（不假设方差相等）
    t_stat, p_value = stats.ttest_ind(exp_data, ctrl_data, equal_var=False)
    
    # 输出结果
    print(f"--- {group_name} ---")
    print(f"实验组均值: {mean_exp:.2f} (std: {std_exp:.2f})")
    print(f"对照组均值: {mean_ctrl:.2f} (std: {std_ctrl:.2f})")
    print(f"t统计量: {t_stat:.2f}, p值: {p_value:.4f}\n")
    
    # 绘制柱状图展示均值差异
    
    plt.figure(figsize=(6,4))
    plt.title(f"{group_name} - active-days-7")
    plt.bar(['exp', 'control'], [mean_exp, mean_ctrl],
            yerr=[std_exp, std_ctrl], capsize=5, color=['blue', 'orange'])
    plt.ylabel("active-days-7")
    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    plt.show()
    

# 对低端设备用户进行分析
analyze_group(low_device_df, "low-device")

# 对年龄敏感用户（30-45 和 60+）进行分析
analyze_group(age_sensitive_df, "age of(30-45, 60+)")