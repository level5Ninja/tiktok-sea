import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from causalml.inference.tree.causal.causaltree import CausalTreeRegressor
from sklearn.tree import _tree

# 定义一个递归函数，遍历决策树并打印每个叶节点的分割条件及其预测的处理效应
def traverse_tree(tree, feature_names, node_id=0, conditions=[]):
    """
    递归遍历决策树，打印每个叶节点的分割条件及其预测值（处理效应）。
    """
    # 如果左子节点等于 TREE_LEAF，则当前节点为叶节点
    if tree.children_left[node_id] == _tree.TREE_LEAF:
        # 获取叶节点预测值（在回归树中，value 通常是 shape=(1, n_outputs)）
        effect = tree.value[node_id][0][0]
        print("Leaf node conditions:", " AND ".join(conditions))
        print("  Predicted treatment effect: {:.4f}".format(effect))
        print("-" * 50)
    else:
        # 当前节点使用的特征及分裂阈值
        feature = feature_names[tree.feature[node_id]]
        threshold = tree.threshold[node_id]
        # 遍历左子树（条件：feature <= threshold）
        traverse_tree(tree, feature_names, tree.children_left[node_id],
                      conditions + [f"{feature} <= {threshold:.2f}"])
        # 遍历右子树（条件：feature > threshold）
        traverse_tree(tree, feature_names, tree.children_right[node_id],
                      conditions + [f"{feature} > {threshold:.2f}"])

# -------------------------------
# 数据加载与预处理
# 请确保 Excel 文件路径正确，文件中包含 'exp_group'、'active_day' 等字段
df = pd.read_excel("/Users/antonio/Desktop/TIKTOK project/AB_test_data.xlsx")

# 将 'exp_group' 转换为二元变量：实验组标记为 1，对照组标记为 0
df['treatment'] = df['exp_group'].apply(lambda x: 1 if x == 'exp_group' else 0)

# 定义因变量 Y（这里使用 active_day 表示 7 日活跃天数）
Y = df['active_day']

# 选择协变量（例如：年龄段、设备等级、区域、性别、媒体来源），并对类别变量做 one-hot 编码
features = ['age_level', 'device_level', 'region', 'gender', 'media_source']
X = pd.get_dummies(df[features], drop_first=True)

# 定义处理变量 T
T = df['treatment']

# 拆分数据为训练集和测试集
X_train, X_test, T_train, T_test, Y_train, Y_test = train_test_split(
    X, T, Y, test_size=0.3, random_state=42)

# -------------------------------
# 初始化并训练因果树模型
ct = CausalTreeRegressor(
    random_state=42,
    max_depth=5,            # 控制树的深度
    min_samples_split=30,   # 允许较低的分裂门槛以捕捉异质性
    min_samples_leaf=30,    # 使叶节点更小，捕捉细分群体
    alpha=0.05,             # 可以尝试不同 alpha 值
    groups_penalty=0.7      # 针对样本不平衡的惩罚力度
)# 注意：fit 方法需要 numpy 数组，因此使用 .values 进行转换
ct.fit(X_train.values, T_train.values, Y_train.values)

# 在测试集上预测个体化处理效应
treatment_effects = ct.predict(X_test.values)
print("预测的个体化处理效应（异质性处理效应）：")
print(treatment_effects)

# -------------------------------
# 遍历内部决策树，提取分割规则（用户画像）
# 注意：ct.tree_ 为内部决策树对象
print("\n因果树分割规则（用户画像）：")
traverse_tree(ct.tree_, list(X.columns))