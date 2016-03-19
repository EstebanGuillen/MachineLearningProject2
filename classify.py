import numpy as np
import math
np.set_printoptions(linewidth=200)

def classify(b):
    numberOfWordsInVocabulary = 61188
    numberOfClasses = 20
    numberOfTestingExamples = 7505

    classProbabilityMatrix = np.zeros((numberOfClasses,))
    wordProbabilityMatrix = np.zeros((numberOfWordsInVocabulary,numberOfClasses))
    wordCountMatrix = np.zeros((numberOfWordsInVocabulary,numberOfClasses))

    rowIndex = 1
    docIdToLabel = {}
    with open('data/train.label', 'r') as trainLabels:
        for line in trainLabels:
            label = int(line.rstrip('\n'))
            docIdToLabel[str(rowIndex)] = label
            classProbabilityMatrix[label-1] = classProbabilityMatrix[label-1] + 1
            rowIndex = rowIndex + 1

    totalTrainingExamples = classProbabilityMatrix.sum()
    classProbabilityMatrix = np.log2(classProbabilityMatrix/totalTrainingExamples)

    with open('data/train.data','r') as trainData:
        for line in trainData:
            values = line.rstrip('\n').split(" ")
            label = docIdToLabel[values[0]]
            wordId = int(values[1])
            wordCount = int(values[2])
            matrixValue = wordCountMatrix[wordId-1,label-1]
            newValue = matrixValue + wordCount
            wordCountMatrix[wordId-1,label-1] = newValue

    #beta = 1.0/numberOfWordsInVocabulary
    beta = b

    vocabSize = numberOfWordsInVocabulary
    for v in range(0,numberOfClasses):
        totalWordsInClassV = wordCountMatrix[:,v].sum()
    
        for w in range(0,numberOfWordsInVocabulary):
            wordId = int(w)
            label = int(v)
            countOfWInClassV = wordCountMatrix[w,v]
            probabilityOfWGivenV = ((countOfWInClassV) + beta)/ (totalWordsInClassV + beta*vocabSize)
            wordProbabilityMatrix[wordId,label] = probabilityOfWGivenV

    wordProbabilityMatrix = np.log2(wordProbabilityMatrix)

    testMatrix = np.zeros((numberOfTestingExamples,numberOfWordsInVocabulary))

    rowIndex = 1
    docIdToTestLabel = {}
    with open('data/test.label', 'r') as testLabels:
        for line in testLabels:
            docIdToTestLabel[str(rowIndex)] = int(line.rstrip('\n'))
            rowIndex = rowIndex + 1

    with open('data/test.data','r') as testData:
        for line in testData:
            values = line.rstrip('\n').split(" ")
            label = docIdToTestLabel[values[0]]
            docId = int(values[0])
            wordId = int(values[1])
            wordCount = int(values[2])
            testMatrix[docId-1,wordId-1] = wordCount

    classifySumMatrix = np.dot(testMatrix,wordProbabilityMatrix)

    classifyProbabilityMatrix = classifySumMatrix + classProbabilityMatrix


    confusionMatrix = np.zeros((numberOfClasses,numberOfClasses))
    errorCount = 0
    correctCount = 0
    for e in range(0,numberOfTestingExamples):
        prediction = np.argmax(classifyProbabilityMatrix[e,:]) + 1
        realLabel = docIdToTestLabel[str(e + 1)]
        confusionMatrix[prediction-1,realLabel-1] = confusionMatrix[prediction-1,realLabel-1] + 1
        if prediction != realLabel:
            errorCount = errorCount + 1
        else:
            correctCount = correctCount + 1
    print(beta)
    print(errorCount)
    print(correctCount/numberOfTestingExamples)
    print(confusionMatrix)


classify(0.00001)
classify(0.0001)
classify(0.001)
classify(0.01)
classify(0.1)
classify(1.0)
