
import math
import heapq
import numpy

class Node:
    def __init__(self, estimate, data, wordIndex, word, infoGain):
        self.estimate = estimate
        self.data = data
        self.wordIndex = wordIndex
        self.word = word
        self.infoGain = infoGain
        self.right = None
        self.left = None
    def __lt__(self, other):
        return self.infoGain < other.infoGain
    
    def __Str__(self):
        return f"{self.word}({self.estimate})"


# Returns the pointEstimate of given data (A or 0)
def pointEstimate(labels, data):
    aCount = 0
    bCount = 0
    for index in data.keys():
        if (index != 0 and labels[index]):
            aCount += 1
        elif (index != 0 and not labels[index]):
            bCount += 1
    if (aCount >= bCount):
        return True # A
    else: 
        return False # B

# Returns the count for A in given data
def getACount(labels, data):
    aCount = 0
    for index in data.keys():
        if (index != 0 and labels[index]):
            aCount += 1
    return aCount

# Gets info of of the given split A and B
def getInfo(a, b):
    total = a + b
    if (total == 0):
        return 1
    elif (a == 0):
        return -1 * b * math.log2(b / total) / total
    elif (b == 0):
        return -1 * a * math.log2(a / total) / total 
    else:
        return -1 * (a * math.log2(a / total) / total + b * math.log2(b / total) / total)

# Gives the best feature in words given method, data and initial info
def getBestFeatureMethod(initialInfo, labels, data, words, halfMethod):
    bestFeature = (0, "", 0)
    for index in words.keys():
        if (index == 0):
            continue
        dataT = {}
        dataF = {}
        for d in data.keys():
            if (data[d][index] == 1 and d != 0 and index != 0):
                dataT.update({d: data[d]})
            elif (data[d][index] == 0 and d != 0 and index != 0):
                dataF.update({d: data[d]})
        aCount = getACount(labels, dataT)
        bCount = len(dataT) - aCount
        infoT = getInfo(aCount, bCount)
        aCount = getACount(labels, dataF)
        bCount = len(dataF) - aCount
        infoF = getInfo(aCount, bCount)
        if (halfMethod):
            infoGain = initialInfo - (0.5 * infoT + 0.5 * infoF)
        else: 
            if (len(data) == 0):
                infoGain = 0
            else:
                infoGain = initialInfo - ((len(dataT) / len(data)) * infoT + (len(dataF) / len(data)) * infoF)
        if (bestFeature[2] < infoGain):
            bestFeature = (index, words[index], infoGain)
    return bestFeature

    
# Returns Decision Tree to determine A or B
def decisionTreeLeaner(labels, data, words, halfMethod):
    infoPQ = []
    estimate = pointEstimate(labels, data)
    aCount = getACount(labels, data)
    bCount = len(data) - aCount 
    initialInfo = getInfo(aCount, bCount)
    bestFeature = getBestFeatureMethod(initialInfo, labels, data, words, halfMethod)
    words.pop(bestFeature[0], None)
    root = Node(estimate, data, bestFeature[0], bestFeature[1], -1 * bestFeature[2])
    heapq.heappush(infoPQ, root)
    nodeCount = 0
    while (nodeCount < 100): 
        leaf = heapq.heappop(infoPQ)
        selectedData = {}
        subtreeT = None
        subtreeF = None
        for i in range(2):
            selectedData = {}
            for index in leaf.data.keys():
                if ((leaf.data[index][leaf.wordIndex] == 1 and i == 0 and index != 0) or (leaf.data[index][leaf.wordIndex] == 0 and i == 1 and index != 0)):
                    selectedData.update({index: leaf.data[index]})
            estimate = pointEstimate(labels, selectedData)
            aCount = getACount(labels, selectedData)
            bCount = len(selectedData) - aCount 
            initialInfo = getInfo(aCount, bCount)
            bestFeature = getBestFeatureMethod(initialInfo, labels, selectedData, words, halfMethod)
            words.pop(bestFeature[0], None)
            subtree = Node(estimate, selectedData, bestFeature[0], bestFeature[1], -1 *  bestFeature[2])
            heapq.heappush(infoPQ, subtree)
            if (i == 0):
                subtreeT = subtree
            else:
                subtreeF = subtree
        leaf.left = subtreeT
        leaf.right = subtreeF
        nodeCount += 1
    return root

# input: binary vector representing words in the document
# classifies input given decistion tree
def classifyExample(input, dt):
    node = dt
    while (node.left != None or node.right != None):
        if (input[node.wordIndex] == 1):
            if (node.left == None):
                return node.estimate
            else:
                node = node.left
        else:
            if (node.right == None):
                return node.estimate
            else:
                node = node.right
    return node.estimate

if __name__ == '__main__':
    # A = True, B = False
    fTrainLabel = open("trainLabel.txt", "r")
    fTrainData = open("trainData.txt", "r")
    fTestLabel = open("testLabel.txt", "r")
    fTestData = open("testData.txt", "r")
    fWords = open("words.txt", "r")
    trainLabels = []
    trainData = {}
    testLabels = []
    testData = {}
    words = {}
    NUMBER_OF_DATA = 1500
    NUMBER_OF_WORDS = 6969

    # offset index by 1
    trainLabels.append(False)
    trainData.update({0: numpy.zeros(NUMBER_OF_WORDS)})
    testLabels.append(False)
    testData.update({0: numpy.zeros(NUMBER_OF_WORDS)})
    words.update({0: ""})

    # read in data
    lines = fTrainLabel.readlines()
    for line in lines:
        if (int(line) == 1):
            trainLabels.append(True)
        else:
            trainLabels.append(False)
    
    lines = fTestLabel.readlines()
    for line in lines:
        if (int(line) == 1):
            testLabels.append(True)
        else:
            testLabels.append(False)
    
    lines = fTrainData.readlines()
    currentDoc = 1
    currentWordList = numpy.zeros(NUMBER_OF_WORDS)
    for line in lines:
        wordList = line.split()
        while (currentDoc != int(wordList[0])):
            trainData.update({currentDoc: currentWordList})
            currentWordList = numpy.zeros(NUMBER_OF_WORDS)
            currentDoc += 1
            if (currentDoc == NUMBER_OF_DATA):
                trainData.update({currentDoc: currentWordList})
                break
        currentWordList[int(wordList[1])] = 1
    
    lines = fTestData.readlines()
    currentDoc = 1
    currentWordList = numpy.zeros(NUMBER_OF_WORDS)
    for line in lines:
        wordList = line.split()
        while (currentDoc != int(wordList[0])):
            testData.update({currentDoc: currentWordList})
            currentWordList = numpy.zeros(NUMBER_OF_WORDS)
            currentDoc += 1
            if (currentDoc == NUMBER_OF_DATA):
                testData.update({currentDoc: currentWordList})
                break
        currentWordList[int(wordList[1])] = 1
    
    lines = fWords.read().splitlines()
    count = 1
    for line in lines:
        words.update({count: line})
        count += 1
    
    dtWeighted = decisionTreeLeaner(trainLabels, trainData, words, False)

    rightCount = 0  
    for index in testData.keys():
        if (index != 1 and classifyExample(testData[index], dtWeighted) == testLabels[index]):
            rightCount += 1
    
    correctPct = rightCount / (len(testData) - 1)
    print(correctPct)


