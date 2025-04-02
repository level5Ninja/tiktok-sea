import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score



df = pd.read_excel("/Users/antonio/Desktop/TIKTOK project/Tiktok SEA user.xlsx")
"""
print("数据概览：")
print(df.head())

print("数据维度: ", df.shape)

print("\n缺失值情况：")
print(df.isnull().sum()) 
"""

# 定义 churn 标签：churn=1 表示流失，churn=0 表示未流失
df['churn'] = df['active_day'].apply(lambda x: 1 if x == 0 else 0)

# 查看流失率基本分布
churn_rate = df['churn'].mean()
print(f"整体流失率：{churn_rate:.2%}")

feature_cols = [
    'region', 'gender', 'os', 'age_level', 'media_source', 'device_level',
    'open_cnt_14', 'vv_cnt_14', 'finish_vv_cnt_14',
    'like_cnt_14', 'comment_cnt_14', 'follow_cnt_14', 'app_usage_duration_30',
]
X = df[feature_cols]
y = df['churn']

cat_cols = ['region','gender','os','age_level','media_source','device_level']



for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))  # 转成str再做编码

#print("\n特征 X 的维度：", X.shape)





X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.3,  # 例如留30%做测试
    random_state=2023,
    stratify=y  # 按照流失标签分层采样，保证流失比例相似
)

print(f"训练集样本数：{X_train.shape[0]}, 测试集样本数：{X_test.shape[0]}")

log_clf = LogisticRegression(max_iter=1000)  # 适当调大迭代次数
log_clf.fit(X_train, y_train)

# 预测概率 (用于计算AUC)
y_pred_proba_log = log_clf.predict_proba(X_test)[:, 1]
# 预测标签 (用于算Precision、Recall)
y_pred_log = log_clf.predict(X_test)


tree_clf = DecisionTreeClassifier(
    max_depth=5, 
    min_samples_split=50, 
    min_samples_leaf=20, 
    random_state=2023
)
tree_clf.fit(X_train, y_train)

y_pred_proba_tree = tree_clf.predict_proba(X_test)[:, 1]
y_pred_tree = tree_clf.predict(X_test)


# 逻辑回归
auc_log = roc_auc_score(y_test, y_pred_proba_log)
precision_log = precision_score(y_test, y_pred_log)
recall_log = recall_score(y_test, y_pred_log)

print("=== 逻辑回归评价 ===")
print(f"AUC: {auc_log:.4f}")
print(f"Precision: {precision_log:.4f}")
print(f"Recall: {recall_log:.4f}")

# 决策树
auc_tree = roc_auc_score(y_test, y_pred_proba_tree)
precision_tree = precision_score(y_test, y_pred_tree)
recall_tree = recall_score(y_test, y_pred_tree)

print("\n=== 决策树评价 ===")
print(f"AUC: {auc_tree:.4f}")
print(f"Precision: {precision_tree:.4f}")
print(f"Recall: {recall_tree:.4f}")

feature_importances_log = pd.DataFrame({
    'Feature': feature_cols,
    'Coefficient': log_clf.coef_[0]
}).sort_values(by='Coefficient', ascending=False)

print("\n逻辑回归特征重要性（回归系数越大 -> 正向影响 churn=1 的概率）：")
print(feature_importances_log)

feature_importances_tree = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': tree_clf.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\n决策树特征重要性：")
print(feature_importances_tree)
