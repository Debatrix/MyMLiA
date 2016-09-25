# -*- coding:utf-8 -*-

from os import listdir
from numpy import *
import operator


def img2vector(filename):
    u"""
     打开指定文件，读取其前32行,并存入数组中返回

     预先为数组分配1*1024空间，并将读入数据线性存入预分配
     数组中
     （可能出现溢出？）
     """
    returnVect = zeros((1, 1024))
    with open(filename) as fr:
        for i in range(32):
            lineStr = fr.readline()
            for j in range(32):
                returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect


def classify0(inX, dataSet, labels, k):
    u'''
     用于分类的输入向量inX,输入的训练样本集dataSet,
     标签向量labels,用于选择最近邻居数目的参数k
     '''
    dataSetSize = dataSet.shape[0]
    # 计算距离
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    #  numpy.argsort() 按照从小到大的顺序返回被排序元素的index
    sortedDistIndicies = distances.argsort()
    classCount = {}
    # 选择距离最小的k个点
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    # 排序
    sortedClassCount = sorted(classCount.iteritems(),\
         key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def handwritingClassTest():
    errorCount = 0.0
    hwLabels = []
    # 获取训练数据目录内容
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)
    # 预分配训练数据内存
    trainingMat = zeros((m, 1024))
    # 遍历训练数据 保存至trainingMat
    for i in range(m):
        # 从文件名解析分类数字
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split(',')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector('trainingDigits/%s' % fileNameStr)
    # 获取测试数据目录内容
    testFileList = listdir('testDigits')
    mTest = len(testFileList)
    # 遍历训练数据 进行分类并计算错误率
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split(',')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUndertest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUndertest, trainingMat, hwLabels, 3)
        print "the classifier came back with: %d, the real answer is: %d" \
            % (classifierResult, classNumStr)
        if classifierResult != classNumStr:
            errorCount += 1
    print "\nthe total number of errors is: %d" % errorCount
    print "\nthe total error rate is: %f" % (errorCount / float(mTest))


if __name__ == '__main__':
    handwritingClassTest()