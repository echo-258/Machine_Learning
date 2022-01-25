# 国家网络安全学院 张家赫 2019302180042

import math
from collections import Counter


def read_data(path):
    with open(path, encoding='UTF-8') as f:
        label = f.readline().strip().split(',')[1:]  # 读取标签行，转换为列表，忽略首项ID
        dataSet = []
        lines = f.readlines()
        for line in lines:
            dataSet.append(line.strip().split(',')[1:])  # 读取数据集
    return label, dataSet


def cal_entropy(dataSet, labelIndex):
    # 此函数用于计算熵，包括全数据集的经验熵、经验条件熵、数据及关于某个特征的熵
    # dataSet传入要计算熵的数据集，labelIndex表示要对哪个特征计算熵（传入特征在原始label中的索引序号。
    dataNum = len(dataSet)
    valueCount = {}
    for data in dataSet:
        value = data[labelIndex]
        if value not in valueCount.keys():
            valueCount[value] = 0
        valueCount[value] += 1  # 采集该特征可能有的所有取值value，以及每种value的取值次数。
    entropy = 0.0
    for key in valueCount:
        p = float(valueCount[key]) / dataNum  # p为一种取值的出现概率
        entropy -= p * math.log(p, 2)  # 依公式计算熵
    return entropy


def split_set(dataSet, featureIndex, value):  # 从一个数据集中分割出某特征为特定value的子集
    ret = []
    for data in dataSet:
        if data[featureIndex] == value:
            ret.append(data)
    return ret


def best_feature(dataSet, feature_used):
    featureNum = len(dataSet[0]) - 1
    H_D = cal_entropy(dataSet, -1)  # 计算全数据集的经验熵
    best_gR_D_A = 0  # 最大信息增益比
    bestFeature = -1  # 最佳特征
    for index in range(featureNum):  # 依次计算每个特征的信息增益比
        if feature_used[index] == 1:
            continue
        values = [data[index] for data in dataSet]
        valueSet = set(values)  # 提取出该特征的所有取值，再去重
        H_D_A = 0.0  # 关于特征A的熵
        for value in valueSet:
            subSet = split_set(dataSet, index, value)  # 求该特征每一种取值的熵，并按取值的出现频率加权求和作为关于该特征的熵
            p = len(subSet) / float(len(dataSet))
            H_D_A += p * cal_entropy(subSet, -1)
        g_D_A = H_D - H_D_A
        gR_D_A = g_D_A / cal_entropy(dataSet, index)  # 信息增益比
        if gR_D_A > best_gR_D_A:
            best_gR_D_A = gR_D_A
            bestFeature = index
    return bestFeature, best_gR_D_A


def create_tree(dataSet, depth, label, feature_used, threshold=0.00001):
    y_list = [data[-1] for data in dataSet]
    if y_list.count(y_list[0]) == len(y_list):  # 如果数据集内所有数据分类相同，则已分类完成
        return y_list[0]
    if 0 not in feature_used:  # 如果所有特征都已用于分类，返回数据集中最多的分类作为此叶节点分类
        return Counter(y_list).most_common()[0]

    bestFeatureIndex, gR_D_A = best_feature(dataSet, feature_used)
    if gR_D_A < threshold:  # 阈值限定
        return Counter(y_list).most_common()[0]
    bestFeature = label[bestFeatureIndex]

    print("depth=", depth, "\tBest Feature is", bestFeature, "\tgR_D_A=", gR_D_A)
    depth += 1

    decisionTree = {bestFeature: {}}  # 用嵌套的字典形式记录树的形状
    feature_used[bestFeatureIndex] = 1  # 用一个数组标记已经使用过的分类

    values = [data[bestFeatureIndex] for data in dataSet]
    valueset = set(values)
    for value in valueset:
        subDataSet = split_set(dataSet, bestFeatureIndex, value)
        decisionTree[bestFeature][value] = create_tree(subDataSet, depth, label, feature_used)
        # 对于当前最佳特征下的每一种取值，递归地寻找下一个最佳特征，构造子树
    return decisionTree


label, dataSet = read_data('data.csv')
# print(label)
# print(dataSet)
feature_used = [0] * len(label)
Decision_Tree = create_tree(dataSet, 0, label, feature_used)
print("Decision Tree:")
print(Decision_Tree)
