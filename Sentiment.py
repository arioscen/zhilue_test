from sklearn import svm
from sklearn.model_selection import cross_val_score
import jieba
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn import neighbors
from sklearn import tree
from sklearn.linear_model import LogisticRegression


# 導入資料,文本進行斷詞
label = []
corpus = []
with open("Ch_trainfile_Sentiment_3000.txt", "r") as f:
    data = f.readlines()
    for line in data:
        try:
            label.append(int(line.split()[0]))
        except ValueError:
            pass
        else:
            cut_list = jieba.cut(line.split()[1])
            corpus.append(",".join(cut_list))

# label 轉爲 ndarry
label_ndarry = np.asarray(label)

# 將文字內容轉換爲向量
vectorizer = CountVectorizer()
corpus_vector = vectorizer.fit_transform(corpus)
corpus_ndarry = np.asarray(corpus_vector.toarray())

# svm 模型
clf = svm.SVC(kernel='linear', C=1)

# 交叉驗證
scores = cross_val_score(clf, corpus_ndarry, label_ndarry, cv=3)

print("SVM Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))  # SVM Accuracy: 0.71 (+/- 0.02)


# knn
knn_model = neighbors.KNeighborsClassifier()
knn_scores = cross_val_score(knn_model, corpus_ndarry, label_ndarry, cv=3)
print("KNN Accuracy: %0.2f (+/- %0.2f)" % (knn_scores.mean(), knn_scores.std() * 2))  # KNN Accuracy: 0.62 (+/- 0.01)

# 決策樹
tree_model = tree.DecisionTreeClassifier()
tree_scores = cross_val_score(tree_model, corpus_ndarry, label_ndarry, cv=3)
print("tree Accuracy: %0.2f (+/- %0.2f)" % (tree_scores.mean(), tree_scores.std() * 2))  # tree Accuracy: 0.67 (+/- 0.01)

# 羅吉斯迴歸
logistic_model = LogisticRegression()
logistic_scores = cross_val_score(logistic_model, corpus_ndarry, label_ndarry, cv=3)
print("logistic Accuracy: %0.2f (+/- %0.2f)" % (logistic_scores.mean(), logistic_scores.std() * 2))  # logistic Accuracy: 0.70 (+/- 0.01)