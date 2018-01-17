import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_auc_score


# 讀取資料
df = pd.read_csv('rs.csv', sep=',', dtype='str')
users = df.user.unique()
items = df.item.unique()

# 設定使用者-商品 DataFrame
user_item_df = pd.DataFrame(0, index=users, columns=items)

# 將資料加入使用者-商品 DataFrame，曾經購買設定爲數值 1
for index, row in df.iterrows():
    user_item_df.loc[row['user'], row['item']] = 1

# ROC 曲線使用 FPR 與 TPR 繪制，所以測試集需加入未購買的情況
all_list = []
for index, row in user_item_df.iterrows():
    for i in range(len(row.index)):
        all_list.append([index, row.index[i], row[i]])

all_df = pd.DataFrame([], columns=['user', 'item', 'value'], dtype='str')

all_df = all_df.append(pd.DataFrame(all_list, columns=['user', 'item', 'value'], dtype='str'))

# 分爲訓練集與測試集
train_data, test_data = train_test_split(all_df, test_size=0.02)

# 設定訓練集 DataFrame
train_df = pd.DataFrame(0, index=users, columns=items)

# 將資料加入訓練集 DataFrame
for index, row in train_data.iterrows():
    train_df.loc[row['user'], row['item']] = int(row['value'])

# 計算物品-物品之間的餘弦相似性
item_similarity = cosine_similarity(train_df.T)

# 物品購買預測
item_prediction = np.array(train_df).dot(item_similarity) / np.array(item_similarity.sum(axis=1))

# 物品購買預測 DataFrame
item_prediction_df = pd.DataFrame(item_prediction, index=users, columns=items)

# 取得與測試集對應的預測值
y_true = []
y_scores = []
for index, row in test_data.iterrows():
    y_true.append(row['value'])
    y_scores.append(item_prediction_df.loc[row['user'], row['item']])

# 計算 AUC 值
print(roc_auc_score(y_true, y_scores))  # 0.65984369663

# 推薦 100 個使用者尚未購買過的商品
results = {}
for idx, row in item_prediction_df.iterrows():
    predict_result = item_prediction_df.loc[idx][train_df.loc[idx] != 1]
    recommender = predict_result.argsort()[:-100:-1]
    results[idx] = recommender.index

# results['userId']
