import time
import numpy as np
from tqdm import tqdm


def read_dataset(path):
    x = []
    y = []
    with open(path) as f:
        lines = f.readlines()
        del (lines[0])
        for line in tqdm(lines):
            sample = line.strip().split(',')
            x.append([int(int(xi) > 128) for xi in sample[1:]])
            y.append(int(sample[0]))
    return x, y


def change_label(y):        # 原来label为0的改为-1，其余均为1
    for i in range(len(y)):
        if y[i] == 0:
            y[i] = -1
        else:
            y[i] = 1


class BoostingTree:
    def __init__(self, train_x, train_y, treeNum=20):
        self.train_data = train_x
        self.train_label = train_y
        self.trainNum, self.featureNum = np.shape(self.train_data)
        self.treeNum = treeNum
        self.D = [1 / self.trainNum] * self.trainNum

        self.alpha = [0] * self.treeNum
        self.e = [1] * self.treeNum
        self.Gx = [[0 for i in range(self.trainNum)] for j in range(self.treeNum)]
        self.feat_index = [-1] * self.treeNum
        self.boundary = [0.0] * self.treeNum
        self.choice = [0] * self.treeNum    # -1表示第feat_index个特征小于等于boundary则为正类，1表示第feat_index个特征大于boundary则为正类
        self.Z = [0] * self.treeNum

    def create_boostingTree(self):
        # 每增加一层数后，当前最终预测结果列表
        finallpredict = [0] * len(self.train_label)

        for m in tqdm(range(self.treeNum)):
            # 得到当前层的提升树
            self.create_onelayer_tree(m)
            # 根据式8.2计算当前层的alpha
            self.alpha[m] = 1 / 2 * np.log((1 - self.e[m]) / self.e[m])
            # 获得当前层的预测结果，用于下一步更新D

            # 更新训练数据集的权值分布
            Zm = 0
            for i in range(self.trainNum):
                Zm += self.D[i] * np.exp(-1 * self.alpha[m] * self.train_label[i] * self.Gx[m][i])
            for i in range(self.trainNum):
                self.D[i] = self.D[i] * np.exp(-1 * self.alpha[m] * self.train_label[i] * self.Gx[m][i]) / Zm
            # self.D = np.multiply(self.D, np.exp(-1 * self.alpha[m] * np.multiply(self.train_label, self.Gx[m]))) / sum(self.D)

            finallpredict += self.alpha[m] * np.array(self.Gx[m])
            # 计算当前最终预测输出与实际标签之间的误差
            error = sum([1 for i in range(self.trainNum) if np.sign(finallpredict[i]) != self.train_label[i]])
            # 计算当前最终误差率
            finallError = error / self.trainNum
            print('iter:%d:%d, single error:%.4f, final error:%.4f' % (m, self.treeNum, self.e[m], finallError))

    def create_onelayer_tree(self, m):
        # 对每一个特征进行遍历，寻找用于划分的最合适的特征
        for i in tqdm(range(self.featureNum), mininterval=1):
            # 因为特征已经经过二值化，只能为0和1，因此分切分时分为-0.5， 0.5， 1.5三挡进行切割
            for bound in [-0.5, 0.5, 1.5]:
                for choi in [-1, 1]:
                    # 按照第i个特征，以值div进行切割，进行当前设置得到的预测和分类错误率
                    pre_result, e = self.calc_err_rate(i, bound, choi)
                    # 如果分类错误率e小于当前最小的e，那么将它作为最小的分类错误率保存
                    if e < self.e[m]:
                        self.e[m] = e
                        self.feat_index[m] = i
                        self.boundary[m] = bound
                        self.choice[m] = choi
                        self.Gx[m] = pre_result

    def calc_err_rate(self, feat_index, bound, choice):
        e = 0
        predict_result = [0] * self.trainNum

        x_feat_index = [self.train_data[i][feat_index] for i in range(self.trainNum)]
        for i in range(self.trainNum):
            if choice == -1:
                if x_feat_index[i] < bound:
                    predict_result[i] = 1
                else:
                    predict_result[i] = -1
            elif choice == 1:
                if x_feat_index[i] >= bound:
                    predict_result[i] = 1
                else:
                    predict_result[i] = -1

        for i in range(self.trainNum):
            if predict_result[i] != self.train_label[i]:
                e += self.D[i]
        return predict_result, e

    def classify(self, x, m):
        x_feat = x[self.feat_index[m]]
        bound = self.boundary[m]
        if self.choice[m] == -1:
            if x_feat < bound:
                return 1
            else:
                return -1
        else:
            if x_feat >= bound:
                return 1
            else:
                return -1

    def test(self, test_x, test_y):
        err_cnt = 0
        test_num = len(test_x)
        # 遍历每一个测试样本
        for i in range(test_num):  # 对每个样本点进行分类
            # 预测结果值，初始为0
            result = 0
            # 依据算法8.1式8.6
            # 预测式子是一个求和式，对于每一层的结果都要进行一次累加
            # 遍历每层的树
            for m in range(self.treeNum):   # 每个弱分类器的分类结果，乘以其系数求和
                result += self.alpha[m] * self.classify(test_x[i], m)
            # 预测结果取sign值，如果大于0 sign为1，反之为0
            if np.sign(result) != test_y[i]:
                err_cnt += 1
        return 1 - err_cnt / test_num


if __name__ == '__main__':
    start = time.time()

    # 读取数据集
    print('read train dataset =======================')
    trainSet_x, trainSet_y = read_dataset('trainSet.csv')  # 读取样本，得到训练集和测试集
    print('read test dataset =======================')
    testSet_x, testSet_y = read_dataset('testSet.csv')
    change_label(trainSet_y)
    change_label(testSet_y)  # label统一改为1或-1

    # 创建提升树
    print('\ntrain boosting tree ======================')
    boostingT_instance = BoostingTree(trainSet_x[:3000], trainSet_y[:3000])
    boostingT_instance.create_boostingTree()

    # 测试
    print('test boosting tree =========================')
    accuracy = boostingT_instance.test(testSet_x[:1000], testSet_y[:1000])
    print('accuracy :%f' % (accuracy * 100), '%')

    # 用时
    print('\noverall time:', time.time() - start)
