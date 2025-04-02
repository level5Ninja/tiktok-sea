import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

# 读取数据文件（请确认文件路径与文件名正确）
df = pd.read_excel("/Users/antonio/Desktop/TIKTOK project/AB_test_data.xlsx")

# 检查数据的基本信息
print("数据列：", df.columns)
print("样本总数：", df.shape[0])

# 定义一个函数，对给定的分层维度进行分析
def subgroup_analysis(df, subgroup_col, metric='active_day'):
    """
    对指定分层维度（例如 region、gender 等）进行实验组和对照组的指标比较，并做t检验。
    """
    # 过滤掉缺失值，并获取该分层所有唯一取值
    subgroups = df[subgroup_col].dropna().unique()
    results = []
    for group in subgroups:
        # 按该分层取值筛选数据
        subgroup_data = df[df[subgroup_col] == group]
        # 根据实验组和对照组筛选数据
        exp_data = subgroup_data[subgroup_data['exp_group'] == 'exp_group'][metric]
        ctrl_data = subgroup_data[subgroup_data['exp_group'] == 'control_group'][metric]
        # 如果样本太少，则跳过
        if len(exp_data) < 5 or len(ctrl_data) < 5:
            continue
        # 计算均值与标准差
        mean_exp = exp_data.mean()
        std_exp = exp_data.std()
        mean_ctrl = ctrl_data.mean()
        std_ctrl = ctrl_data.std()
        # 进行独立样本t检验（不假设方差相等）
        t_stat, p_value = stats.ttest_ind(exp_data, ctrl_data, equal_var=False)
        results.append({
            subgroup_col: group,
            'exp_n': len(exp_data),
            'ctrl_n': len(ctrl_data),
            'mean_exp': mean_exp,
            'std_exp': std_exp,
            'mean_ctrl': mean_ctrl,
            'std_ctrl': std_ctrl,
            't_stat': t_stat,
            'p_value': p_value,
            'diff': mean_exp - mean_ctrl,
            'relative_diff': (mean_exp - mean_ctrl) / mean_ctrl if mean_ctrl != 0 else None,
        })
    return pd.DataFrame(results)

# 设定需要进行下探分析的分层维度
subgroup_columns = ['region', 'gender', 'device_level', 'media_source', 'age_level']

# 针对每个分层维度，进行下探分析，并输出结果

for col in subgroup_columns:
    print(f"\n===== 分层分析：{col} =====")
    res = subgroup_analysis(df, col, metric='active_day')
    if res.empty:
        print(f"{col} 分层样本太少，无法进行有效统计")
    else:
        print(res)
        # 可选：可视化每个分组的均值差异
        plt.figure(figsize=(10, 6))
        plt.title(f"{col} dif of 7days AU")
        plt.bar(res[col].astype(str), res['diff'])
        plt.xlabel(col)
        plt.ylabel("dif in mean")
        plt.axhline(0, color='black', linestyle='--', linewidth=1)
        plt.show()"
        