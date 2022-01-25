import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

sample = np.array([[34, 30, 1],
                 [32, 6, 0],
                 [30, 20, 0],
                 [30, 26, 1],
                 [5, 25, 0],
                 [16, 53, 0],
                 [8, 19, 0],
                 [20, 47, 0],
                 [24, 32, 1],
                 [2, 17, 1],
                 [9, 24, 1],
                 [47, 17, 0],
                 [10, 40, 1],
                 [23, 34, 1],
                 [32, 35, 1],
                 [23, 16, 1],
                 [26, 45, 0],
                 [16, 5, 0],
                 [34, 52, 0],
                 [37, 25, 1]])
featureSet = sample[:, 0:2]
labelSet = sample[:, -1]

classifier = (KNeighborsClassifier(n_neighbors=1, n_jobs=-1), KNeighborsClassifier(n_neighbors=2, n_jobs=-1))
models = (clf.fit(featureSet, labelSet) for clf in classifier)  # 使用sklearn建立模型

x_min, x_max = featureSet[:, 0].min() - 1, featureSet[:, 0].max() + 1
y_min, y_max = featureSet[:, 1].min() - 1, featureSet[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.2), np.arange(y_min, y_max, 0.2))

titles = ('k=1', 'k=2')
fig = plt.figure(figsize=(10, 5))
for clf, title, ax in zip(models, titles, fig.subplots(1, 2).flatten()):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    colors = ('blue', 'red')
    cmap = ListedColormap(colors[:len(np.unique(Z))])
    ax.contourf(xx, yy, Z, cmap=cmap, alpha=0.5)  # 标记分区
    ax.scatter(featureSet[:, 0], featureSet[:, 1], c=labelSet, cmap=cmap)  # 标记样本点
    ax.set_title(title)

plt.savefig("cmp_k_1_2")

'''
运行结论：
k取值较小时，预测结果对近邻的实例点更敏感，学习的近似误差小；整体模型复杂，决策边界更细致。
k取值较大时，与输入实例较远的训练实例也会对预测其作用，学习的估计误差小；整体模型简单，决策边界更粗略；
'''
