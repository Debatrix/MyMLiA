# -*- coding:utf-8 -*-
from numpy import *
u"""
 支持向量机 采用SMO优化
 """


def loadDataSet(fileName):
    u"""
     数据载入 分为数据列表和标签列表
     """
    dataMat = []
    labelMat = []
    with open(fileName) as fr:
        for line in fr.readlines():
            lineArr = line.strip().split('\t')
            dataMat.append([float(lineArr[0]), float(lineArr[1])])
            labelMat.append(float(lineArr[2]))
    return dataMat, labelMat


def selectJrand(i, m):
    u"""
     随机选择两个alpha
     """
    j = 1
    while (j == i):
        j = int(random.uniform(0, m))
    return j


def clipAlpha(aj, H, L):
    u"""
     确保aj在H和L之间
     """
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    u"""
     简化版SMO算法 

     参数：
     dataMatIn    数据集
     classLabels  标签集
     C            松弛变量  用于控制“最大化间隔”和
                             “保证大部分点的函数间隔小于1.0”
     toler        容错率
     maxIter      退出前最大迭代数

     工作原理：

     SMO算法的目标是求出一系列alpha和b 通过alpha求出权重向量w
     每次循环中选出两个alpha进行优化处理 增大其中一个减小另一个
     优化条件有两个： 1.两个alpha必须在间隔边界外 2.这两个alpha
     还没有经过区间化处理或者不在边界上
     
     其伪代码如下

     创建一个alpha向量并将其初始化为0向量
     当迭代次数小于最大迭代次数时（外循环）：
        对数据集中的每个数据向量（内循环）：
            如果该数据可以被优化：
                随机选择另一个数据向量
                同时优化这个数据向量
                如果该向量不能被优化，退出内循环
        如果所有向量都没被优化，增加迭代数目，继续下一次循环
     """
    # 初始化
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    b = 0
    m, n = shape(dataMatrix)
    alphas = mat(zeros((m, 1)))
    iter = 0
    while (iter < maxIter):
        alphaPairsChanged = 0
        for i in range(m):
            # fXi 预测的类别 Ei 误差
            fXi = float(multiply(alphas, labelMat).T *
                        (dataMatrix * dataMatrix[i, :].T)) + b
            Ei = fXi - float(labelMat[i])
            # 间隔太大（正负）同时alpha在边界内（出界会被调整至边界）则进行优化
            # 在边界上的alpha不值得优化
            if ((labelMat[i] * Ei < -toler) and (alphas[i] < C)) or \
                    ((labelMat[i] * Ei > toler) and (alphas[i] > 0)):
                j = selectJrand(i, m)
                fXj = float(multiply(alphas, labelMat).T *
                            (dataMatrix * dataMatrix[j, :].T)) + b
                Ej = fXj - float(labelMat[j])
                # 为备份分配内存
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                # H和L用于将alpha保持在0和C之间
                if (labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L == H:
                    print "L==H"
                    continue
                # 最优修改量
                eta = 2.0 * dataMatrix[i, :] * dataMatrix[j, :].T - \
                    dataMatrix[i, :] * dataMatrix[i, :].T - \
                    dataMatrix[j, :] * dataMatrix[j, :].T
                if eta >= 0:
                    print "eta>=0"
                    continue
                alphas[j] -= labelMat[j] * (Ei - Ej) / eta
                alphas[j] = clipAlpha(alphas[j], H, L)
                if (abs(alphas[j] - alphaJold) < 0.00001):
                    print "j not moving enough"
                    continue
                alphas[i] += labelMat[j] * \
                    labelMat[i] * (alphaJold - alphas[j])
                b1 = b - Ei - labelMat[i] * (alphas[i] - alphaIold) * \
                    dataMatrix[i, :] * dataMatrix[i, :].T -\
                    labelMat[j] * (alphas[j] - alphaJold) * \
                    dataMatrix[i, :] * dataMatrix[j, :].T
                b2 = b - Ej - labelMat[i] * (alphas[i] - alphaIold) * \
                    dataMatrix[i, :] * dataMatrix[j, :].T - \
                    labelMat[j] * (alphas[j] - alphaJold) * \
                    dataMatrix[j, :] * dataMatrix[j, :].T
                if (0 < alphas[i]) and (C > alphas[i]):
                    b = b1
                elif (0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                alphaPairsChanged += 1
                print "iter: %d i:%d, pairs changed %d" % \
                    (iter, i, alphaPairsChanged)
        if (alphaPairsChanged == 0):
            iter += 1
        else:
            iter = 0
        print "iteration number: %d" % iter
    return b, alphas


def test():
    dataArr, labelArr = loadDataSet("testSet.txt")
    b, alphas = smoSimple(dataArr, labelArr, 0.6, 0.001, 40)
    print b, '\n', alphas[alphas > 0]


if __name__ == '__main__':
    test()
