# -*- coding:utf-8 -*-
from numpy import *
"""
总的来说就是利用贝叶斯公式计算元素属于某一分类的概率 将元素归纳到某一分类的方式
适合处理文本分类之类的离散的数据

计算并非完全符合贝叶斯公式 
因为除以（或者取对数后减去）元素出现的概率这个相同的值在这里没有意义
"""


def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']
                   ]
    classVec = [0, 1, 0, 1, 0, 1]    # 1 is abusive, 0 not
    return postingList, classVec


def createVocabList(dataSet):
    u"""
     创建一个包含在所有文档中出现的不重复词的列表
     """
    vocabSet = set([])
    for document in dataSet:
        # 集合或运算也就是取其并集 就是求和
        vocabSet = vocabSet | set(document)
    return list(vocabSet)


def setOfWords2Vec(vocabList, inputSet):
    u"""
     确定输入集合中的单词是否在词汇表中出现 输出一个与词汇表等长的
     向量 出现的单词在输出向量对应位置置1

     参数：
     vocabList 词汇表
     inputSet 输入集合
     """
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("The word: %s is not in my Vocabulary!" % word)
    return returnVec


def trainNB0(trainMatrix, trainCategory):
    u"""
     朴素贝叶斯训练器 
     计算样本中出现的不同词汇出现的条件概率与不同类型文章出现的概率

     参数：
     trainMatrix 词汇表 文章出现的单词在词汇表对应位置置1
     trainCategory 文章分类数组

     主要就是计算条件概率
     对于非二类分类问题需要修改
     """
    #统计文章数目，单词表长度
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    #计算不同类型文章出现的概率
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    #初始化分子分母变量 
    #为防止某个概率为0之后的乘积均为零 将所有词的出现次数初始化为1 分母初始化为2
    p0Num = ones(numWords)
    p1Num = ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0
    #分别统计每个词在某类文章中的出现次数（+1）与属于该类文章总的词汇数（+2）
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    #计算条件概率 并取对数（保证精度）
    p1Vect = log(p1Num / p1Denom)
    p0Vect = log(p0Num / p0Denom)
    return p0Vect, p1Vect, pAbusive


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    u"""
     贝叶斯分类器 
     将词汇表与条件概率向量相乘后与文章分类概率的对数值求和
     
     参数：
     vec2Classify 文章词汇表
     p0Vec p0的条件概率
     p1Vec p1的条件概率
     pClass1 文章的分类概率
     """
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec


def textParse(bagString):
    import re
    listOfTokens = re.split(r'\W*', bagString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]


def spamTest():
    docList = []
    classList = []
    fullText = []
    for i in range(1, 26):
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    trainingSet = range(50)
    testSet = []
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))
    errorCount = 0.0
    for docIndex in testSet:
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])
        flag = classifyNB(array(wordVector), p0V, p1V, pSpam)
        if flag != classList[docIndex]:
            errorCount += 1
            print "classification error", docList[docIndex]
    print 'The error rate is:', float(errorCount) / len(testSet)


def testingNB():
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNB0(array(trainMat), array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print testEntry, ' classified as: ', classifyNB(thisDoc, p0V, p1V, pAb)
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print testEntry, ' classified as: ', classifyNB(thisDoc, p0V, p1V, pAb)
    testEntry = ['love', 'my', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print testEntry, ' classified as: ', classifyNB(thisDoc, p0V, p1V, pAb)


if __name__ == '__main__':
    testingNB()
