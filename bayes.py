import numpy as np

# 任务：
# 给定句子，以及是否有侮辱性，判断输入的句子是否带有侮辱性词汇
# 1具有侮辱性，0不具有侮辱性
postingList = [
    ['my', 'dog', 'has', 'flea', 'problems', 'help','please'],
    ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
    ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
    ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
    ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
    ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']
]

classVec = [0, 1, 0, 1, 0, 1]

# 将所有句子中出现的词汇合并为一个列表
def createVocabList(dataSet):
    vocabSet = set()
    for documet in dataSet:
        vocabSet = vocabSet | set(documet)
    return list(vocabSet)

# 将句子中单词在列别表中是否出现打表
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return returnVec

def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    # 拉普拉斯平滑
    p0num = np.ones(numWords)
    p1num = np.ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1num += trainMatrix[i]
            p1Denom += np.sum(trainMatrix[i])
        else:
            p0num += trainMatrix[i]
            p0Denom += np.sum(trainMatrix[i])
    # 防止下溢处理
    p1Vect = np.log(p1num / p1Denom)
    p0Vect = np.log(p0num / p0Denom)
    return p0Vect, p1Vect, pAbusive

# 生成输入的数字矩阵
myVocabList = createVocabList(postingList)
trainMat = []
for postinDoc in postingList:
    trainMat.append(setOfWords2Vec(myVocabList, postinDoc))

# 使用贝叶斯分类
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + np.log(1-pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

thisDoc = setOfWords2Vec(myVocabList, ['stupid', 'garbage'])
result = classifyNB(thisDoc, *trainNB0(trainMat, classVec))
print(result)
thisDoc = setOfWords2Vec(myVocabList, ['love', 'my', 'dalmation'])
result = classifyNB(thisDoc, *trainNB0(trainMat, classVec))
print(result)