import matplotlib.pyplot as plt
import numpy as np

result = {'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}}

def getNumLeafs(myTreee):
    numLeafs = 0
    firstStr = myTreee.keys()[0]
    secondDict = myTreee[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs

def getTreeDepth(myTreee):
    maxDepth = 0
    firstStr = myTreee.keys()[0]
    secondDict = myTreee[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            thisDepth = 1 + getTreeDepth(secondDict)
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth