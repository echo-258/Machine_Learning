import time
import numpy as np
import math
import random
from tqdm import tqdm


def read_dataset(path):
    x = []
    y = []
    with open(path) as f:
        lines = f.readlines()
        del (lines[0])
        for line in tqdm(lines):
            sample = line.strip().split(',')
            x.append([int(xi) / 255 for xi in sample[1:]])
            y.append(int(sample[0]))
    return x, y


def change_label(y):        # 原来label为0的改为-1，其余均为1
    for i in range(len(y)):
        if y[i] == 0:
            y[i] = -1
        else:
            y[i] = 1


def dim_reduction(x):
    ret = []
    size = int(len(x[0]) ** 0.5)
    partial_size = 4
    for sample in tqdm(x):
        sample_ret = []
        for i in range(size // partial_size):
            for j in range(size // partial_size):
                sum = 0
                for a in range(partial_size):
                    for b in range(partial_size):
                        sum += sample[(i * partial_size + a) * size + (j * partial_size + b)]
                sum = sum / (partial_size * partial_size)
                sample_ret.append(sum)
        ret.append(sample_ret)
    return ret


class SVM:
    def __init__(self, trainDataList, trainLabelList, sigma=10, C=5000, epsilon=0.001):
        self.train_data = np.mat(trainDataList)   # 训练数据集，作为一个矩阵
        self.train_label = trainLabelList   # 训练标签集
        self.m, self.n = np.shape(self.train_data)    # m是训练集数量，n是样本特征数目
        self.sigma = sigma      # 高斯核分母中的σ
        self.C = C          # 软间隔中的惩罚参数
        self.epsilon = epsilon  # 判断是否满足KTT条件的精度范围
        self.K = self.init_kernel()     # 训练样本点两两之间的核函数计算结果
        self.b = 0      # SVM中的阈值b
        self.alpha = [0] * self.train_data.shape[0]   # α 长度为训练集数目
        self.E = [-1 * self.train_label[i] for i in range(self.m)]     # SMO运算过程中的E
        self.supportVecIndex = []       # 训练样本点中支持向量的索引集合

    def calc_kernel(self, x1, x2):  # 计算样本点x1和x2的高斯核函数值
        result = (x1 - x2) * (x1 - x2).T
        result = np.exp(-1 * result / (2 * self.sigma ** 2))
        return np.exp(result)

    def init_kernel(self):      # 初始化各训练样本点之间的核函数，这里采用高斯核函数
        # 核函数是对每两个xi和xj计算的，训练样本点共有m个，故使用m*m矩阵存放所有计算结果。此结果只由训练集决定，再训练过程中不改变
        print('\t\tinitiate kernel ----------------')

        k = [[0 for i in range(self.m)] for j in range(self.m)]
        for i in tqdm(range(self.m)):
            xi = self.train_data[i, :]    # 也就是书上公式中的x
            for j in range(i, self.m):
                xj = self.train_data[j, :]    # 也就是书上公式中的z
                result = self.calc_kernel(xi, xj)
                k[i][j] = result
                k[j][i] = result    # K(xi, xj)与K(xj, xi)的结果是一样的
        return k

    def satisfy_KKT(self, i):   # 判断第i个α是否满足KKT条件
        gxi = self.calc_gxi(i)
        yi = self.train_label[i]

        if (math.fabs(self.alpha[i]) < self.epsilon) and (yi * gxi >= 1):
            return True
        elif (math.fabs(self.alpha[i] - self.C) < self.epsilon) and (yi * gxi <= 1):
            return True
        elif (self.alpha[i] > -self.epsilon) and (self.alpha[i] < (self.C + self.epsilon)) \
                and (math.fabs(yi * gxi - 1) < self.epsilon):
            return True
        return False

    def calc_gxi(self, i):  # 计算g(xi)   这里只取支持向量进行计算，否则奇慢无比 十分钟完成三轮迭代
        gxi = 0
        # 先取非零α的索引，对应支持向量
        index = [i for i, alpha in enumerate(self.alpha) if alpha != 0]
        for j in index:
            gxi += self.alpha[j] * self.train_label[j] * self.K[j][i]
        gxi += self.b
        return gxi

    def select_j(self, E1, i):  # 选取第二个α
        E2 = 0
        maxE1_E2 = 0
        max_dif_j = -1

        # 优先选择经历过修改的E
        modified_E_index = [i for i, Ei in enumerate(self.E) if Ei != -1 * self.train_label[i]]
        if len(modified_E_index) == 0:      # 没有经历过修改的E（刚刚开始训练时）
            max_dif_j = i
            while max_dif_j == i:
                max_dif_j = int(random.uniform(0, self.m))   # 随机选择一个α作为α2，且不能与α1是同一个
            E2 = self.calc_gxi(max_dif_j) - self.train_label[max_dif_j]
        else:
            for j in modified_E_index:
                E2_tmp = self.calc_gxi(j) - self.train_label[j]
                if math.fabs(E1 - E2_tmp) > maxE1_E2:   # 选择|E1-E2|最大的作为E2
                    maxE1_E2 = math.fabs(E1 - E2_tmp)
                    E2 = E2_tmp
                    max_dif_j = j

        return E2, max_dif_j

    def train(self, iteration=10):
        for cur_iter in range(iteration):
            print('\t\titeration: %d / %d ----------------' % (cur_iter, iteration))
            time.sleep(0.5)

            for i in tqdm(range(self.m)):   # 外层循环：选择第一个变量
                if not self.satisfy_KKT(i):   # 如果下标为i的第一个α不满足KKT条件，则进行优化
                    # 选择第二个α。首先需要计算E1
                    E1 = self.calc_gxi(i) - self.train_label[i]
                    E2, j = self.select_j(E1, i)   # 选择第2个α，其索引为j

                    y1 = self.train_label[i]
                    y2 = self.train_label[j]
                    a1_old = self.alpha[i]
                    a2_old = self.alpha[j]

                    if y1 != y2:  # 对于y1和y2是否相同的情况，需要分别讨论
                        L = max(0, a2_old - a1_old)
                        H = min(self.C, self.C + a2_old - a1_old)
                    else:
                        L = max(0, a2_old + a1_old - self.C)
                        H = min(self.C, a2_old + a1_old)
                    if L == H:
                        continue  # 二者相等则无法优化，进入下一次对α的选择

                    # 先在未经剪辑的情况下，计算α2的新值
                    eta = self.K[i][i] + self.K[j][j] - 2 * self.K[i][j]  # 分母η
                    a2_new = a2_old + y2 * (E1 - E2) / eta
                    # 按照前面的约束范围，对α2进行剪辑
                    if a2_new < L:
                        a2_new = L
                    elif a2_new > H:
                        a2_new = H
                    # 利用更新后的α的值更新α1
                    a1_new = a1_old + y1 * y2 * (a2_old - a2_new)

                    # 更新阈值b
                    b1_new = self.b - E1 - y1 * self.K[i][i] * (a1_new - a1_old) - y2 * self.K[j][i] * (a2_new - a2_old)
                    b2_new = self.b - E2 - y1 * self.K[i][j] * (a1_new - a1_old) - y2 * self.K[j][j] * (a2_new - a2_old)
                    if (a1_new > 0) and (a1_new < self.C):  # 依据α1和α2的值范围确定新b
                        b_new = b1_new
                    elif (a2_new > 0) and (a2_new < self.C):
                        b_new = b2_new
                    else:
                        b_new = (b1_new + b2_new) / 2

                    # 将更新后的alpha、b、E记录在SVM的属性中，供下次更新使用
                    self.alpha[i] = a1_new
                    self.alpha[j] = a2_new
                    self.b = b_new
                    self.E[i] = self.calc_gxi(i) - self.train_label[i]
                    self.E[j] = self.calc_gxi(j) - self.train_label[j]

        # 迭代优化过程结束后，统计支持向量，即对应α>0的样本点，记录在SVM类的属性中
        for i in range(self.m):
            if self.alpha[i] > 0:
                self.supportVecIndex.append(i)

    def predict(self, x):
        result = 0
        for i in self.supportVecIndex:
            # 非支持向量的α值为0，不需要计算；只需要对所有支持向量计算αi * yi * x * K(xi, x)，然后求和即可
            kernel = self.calc_kernel(self.train_data[i, :], np.mat(x))
            result += self.alpha[i] * self.train_label[i] * kernel
        result += self.b
        return np.sign(result)

    def test(self, testDataList, testLabelList):
        errorCnt = 0        # 误分类样本点计数
        for i in tqdm(range(len(testDataList))):
            result = self.predict(testDataList[i])
            if result != testLabelList[i]:
                errorCnt += 1       # 出现误分类
        return 1 - errorCnt / len(testDataList)


if __name__ == '__main__':
    train_num = 2000
    test_num = 1000
    start = time.time()

    # 读取数据集
    print('read train dataset =======================')
    trainSet_x, trainSet_y = read_dataset('trainSet.csv')  # 读取样本，得到训练集和测试集
    print('read test dataset =======================')
    testSet_x, testSet_y = read_dataset('testSet.csv')
    print('dimension reduction for samples =======================')
    trainSet_x = dim_reduction(trainSet_x)
    testSet_x = dim_reduction(testSet_x)
    print('change label for samples =======================')
    change_label(trainSet_y)
    change_label(testSet_y)  # label统一改为1或-1

    # 初始化SVM类
    print('\ninitiate SVM =======================')
    svm = SVM(trainSet_x[:train_num], trainSet_y[:train_num])

    # 训练SVM模型
    print('\ntrain SVM =======================')
    svm.train()

    # 测试SVM模型
    print('\ntest SVM =====================')
    accuracy = svm.test(testSet_x[:test_num], testSet_y[:test_num])
    print('accuracy :%f' % (accuracy * 100), '%')

    # 用时
    print('\noverall time:', time.time() - start)