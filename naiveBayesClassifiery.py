import numpy as np
import math

np.set_printoptions(linewidth=200)



def classify(beta, numberOfWordsInVocabulary, numberOfClasses, numberOfTestingExamples, numberOfTrainingExamples, printTop100AndBottom100):
    
    classProbabilityMatrix = np.zeros((numberOfClasses,))
    wordProbabilityMatrix = np.zeros((numberOfWordsInVocabulary,numberOfClasses))
    wordProbabilityLogMatrix = np.zeros((numberOfWordsInVocabulary,numberOfClasses))
    wordCountMatrix = np.zeros((numberOfWordsInVocabulary,numberOfClasses))
    
    wordToDocumentCount = {}
    
    
    rowIndex = 1
    docIdToLabel = {}
    with open('data/train.label', 'r') as trainLabels:
        for line in trainLabels:
            label = int(line.rstrip('\n'))
            docIdToLabel[str(rowIndex)] = label
            classProbabilityMatrix[label-1] = classProbabilityMatrix[label-1] + 1.0
            rowIndex = rowIndex + 1

    totalTrainingExamples = classProbabilityMatrix.sum()
    classProbabilityMatrix = classProbabilityMatrix/totalTrainingExamples
    classProbabilityLogMatrix = np.log2(classProbabilityMatrix)


    vocabList = []
    with open('data/vocabulary.txt','r') as vocab:
        for line in vocab:
            value = line.rstrip('\n')
            vocabList.append(value)

    
    with open('data/train.data','r') as trainData:
        for line in trainData:
            values = line.rstrip('\n').split(" ")
            label = docIdToLabel[values[0]]
            wordId = int(values[1])
            wordCount = int(values[2])
            matrixValue = wordCountMatrix[wordId-1,label-1]
            newValue = matrixValue + wordCount
            wordCountMatrix[wordId-1,label-1] = newValue
            wordToDocumentCount[wordId-1] = 1



    vocabSize = numberOfWordsInVocabulary
    for v in range(0,numberOfClasses):
        totalWordsInClassV = wordCountMatrix[:,v].sum()
        
        for w in range(0,numberOfWordsInVocabulary):
            wordId = int(w)
            label = int(v)
            countOfWInClassV = wordCountMatrix[w,v]
            probabilityOfWGivenV = ((countOfWInClassV) + beta)/ (totalWordsInClassV + beta*vocabSize)
            wordProbabilityMatrix[wordId,label] = probabilityOfWGivenV





    wordProbabilityLogMatrix = np.log2(wordProbabilityMatrix)
    

    if printTop100AndBottom100:
        informationGainVector = np.zeros((numberOfWordsInVocabulary))


        for w in wordToDocumentCount.keys():
            sum = 0
        
            numberOfDocumentsContainingW = wordToDocumentCount[w]
            probabilityADocumentContainsW = numberOfDocumentsContainingW  / numberOfTrainingExamples
            classEntropySum = 0
            wordEntropyPosSum = 0
            wordEntropyNegSum = 0
            for v in range(0,numberOfClasses):
                probabilityOfWGivenClass = wordProbabilityMatrix[w,v]
               
                probabilityOfClass = classProbabilityMatrix[v]
            
                classEntropy = (probabilityOfClass * math.log(probabilityOfClass,2))
                classEntropySum = classEntropySum + classEntropy
            
                wordEntropyPos = probabilityOfWGivenClass * math.log(probabilityOfWGivenClass,2)
                wordEntropyPosSum = wordEntropyPosSum + wordEntropyPos
            
                wordEntropyNeg = (1.0-probabilityOfWGivenClass) * math.log( (1.0 - probabilityOfWGivenClass), 2 )
                wordEntropyNegSum = wordEntropyNegSum + wordEntropyNeg
            

            sum = -(classEntropySum) + probabilityADocumentContainsW * wordEntropyPosSum + (1.0-probabilityADocumentContainsW) * wordEntropyNegSum
        
            informationGainVector[w] = sum


        topNWords = np.argpartition(informationGainVector, -100)[-100:]
        for w in topNWords:
            print(vocabList[w])
        bottomNWords = np.argpartition(informationGainVector, 100)[:100]



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









    classifySumMatrix = np.dot(testMatrix, wordProbabilityLogMatrix)
    
    classifyProbabilityMatrix = classifySumMatrix + classProbabilityLogMatrix


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
    print("Accuracy",correctCount/numberOfTestingExamples)
    print(confusionMatrix)
    return correctCount/numberOfTestingExamples





classify(0.001, 61188,20,7505,11269,False)

classify(0.01, 61188,20,7505,11269,False)

classify(0.1, 61188,20,7505,11269,False)

classify(1.0, 61188,20,7505,11269,False)

classify(1.0, 61188,20,7505,11269,True)




