import numpy as np
import math
#import matplotlib.pyplot as plt

np.set_printoptions(linewidth=200)


#important note: the training and testing data is indexed starting at 1
#                so wordId and docId needed to be subracted by 1 to fit nicely into a matrix (zero indexed)

#function to classify 20 newsgroup dataset given a beta value (hallucinated values)
#returns the accuracy of the classifier on the testing data
#prints a confusion matrix to the console
def classify(beta, numberOfWordsInVocabulary, numberOfClasses, numberOfTestingExamples, numberOfTrainingExamples, printTop100InformationGainWords):
    
    #defining some matrices to be used later in the code for holding data and performing linear algebra operations
    classProbabilityMatrix = np.zeros((numberOfClasses,))
    wordProbabilityMatrix = np.zeros((numberOfWordsInVocabulary,numberOfClasses))
    wordProbabilityLogMatrix = np.zeros((numberOfWordsInVocabulary,numberOfClasses))
    wordCountMatrix = np.zeros((numberOfWordsInVocabulary,numberOfClasses))
    
    #dictionary to map how many documents a word appears in, will be used for the information gain calculation
    wordToDocumentCount = {}
    
    #read in the listing of training labels
    rowIndex = 1
    docIdToLabel = {}
    with open('data/train.label', 'r') as trainLabels:
        for line in trainLabels:
            label = int(line.rstrip('\n'))
            docIdToLabel[str(rowIndex)] = label
            classProbabilityMatrix[label-1] = classProbabilityMatrix[label-1] + 1.0
            rowIndex = rowIndex + 1

    totalTrainingExamples = classProbabilityMatrix.sum()
    
    #the probability (using MLE) of a document falling into a given class is the number of 
    #documents in a class divided by the total number of documents
    classProbabilityMatrix = classProbabilityMatrix/totalTrainingExamples
    #to help fight against underflow we will be using the log2 of our probability values
    classProbabilityLogMatrix = np.log2(classProbabilityMatrix)

    #read through the vocabulary listing and store in a list
    vocabList = []
    with open('data/vocabulary.txt','r') as vocab:
        for line in vocab:
            value = line.rstrip('\n')
            vocabList.append(value)

    #read through the train data, file is in the format "docId wordId wordCount"
    with open('data/train.data','r') as trainData:
        for line in trainData:
            values = line.rstrip('\n').split(" ")
            label = docIdToLabel[values[0]]
            wordId = int(values[1])
            wordCount = int(values[2])
            matrixValue = wordCountMatrix[wordId-1,label-1]
            newValue = matrixValue + wordCount
            #build the word count matrix (being careful with the indexes)
            #the wordCountMatrix is needed to calculate the wordProbabilityMatrix below
            wordCountMatrix[wordId-1,label-1] = newValue
            wordToDocumentCount[wordId-1] = 1


    #create the wordProbabilityMaxtix (probability of a word given the class)
    #the wordProbabilityMatrix will have shape [numberOfWordsInvocabulary X numberOfClasses]
    vocabSize = numberOfWordsInVocabulary
    for v in range(0,numberOfClasses):
        totalWordsInClassV = wordCountMatrix[:,v].sum()
        
        for w in range(0,numberOfWordsInVocabulary):
            wordId = int(w)
            label = int(v)
            countOfWInClassV = wordCountMatrix[w,v]
            #Using MAP estimates to calculate probabilities
            probabilityOfWGivenV = ((countOfWInClassV) + beta)/ (totalWordsInClassV + beta*vocabSize)
            wordProbabilityMatrix[wordId,label] = probabilityOfWGivenV

    #again we will use the log2 of the probabilities to help fight against underflow
    wordProbabilityLogMatrix = np.log2(wordProbabilityMatrix)
    

    #start information gain code
    #the following block of code calcualtes the infomation gain for each word and print the top 100
    #source of equation: http://www.time.mk/trajkovski/thesis/text-class.pdf (section 2.2)
    if printTop100InformationGainWords:
        informationGainVector = np.zeros((numberOfWordsInVocabulary))
        #only loop over words that were encountered in training data 
        #(some words in vocabulary.txt didn't show up anywhere)
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
            

            sum = -(classEntropySum) + probabilityADocumentContainsW * wordEntropyPosSum + \
                                       (1.0-probabilityADocumentContainsW) * wordEntropyNegSum
        
            informationGainVector[w] = sum

        topNWords = np.argpartition(informationGainVector, -100)[-100:]
        topList = []
        for w in topNWords:
            topList.append(vocabList[w])
        print("")
        print("")
        print("TOP 100 WORDS")
        print(topList)
        print("")
        print("")
    #end information gain code

    
    #matrix will hold the test data and will be used in the classification operation
    testMatrix = np.zeros((numberOfTestingExamples,numberOfWordsInVocabulary))
    
    #read through the test labels and save to a dictionary, values will be used to calculate accuracy
    rowIndex = 1
    docIdToTestLabel = {}
    with open('data/test.label', 'r') as testLabels:
        for line in testLabels:
            docIdToTestLabel[str(rowIndex)] = int(line.rstrip('\n'))
            rowIndex = rowIndex + 1

    #read through the test data and populate the testMatrix with word counts
    with open('data/test.data','r') as testData:
        for line in testData:
            values = line.rstrip('\n').split(" ")
            label = docIdToTestLabel[values[0]]
            docId = int(values[0])
            wordId = int(values[1])
            wordCount = int(values[2])
            testMatrix[docId-1,wordId-1] = wordCount


    #part of classify calculation 
    #  Sum of Xi * log2(P(Xi|Yk)) -
    classifySumMatrix = np.dot(testMatrix, wordProbabilityLogMatrix)
    
    #finishing out the classify calculation, adding in the Class probability P(Yk)
    classifyProbabilityMatrix = classifySumMatrix + classProbabilityLogMatrix

    #confusionMatrix to show a nice visualization of our classifcation accuracy
    confusionMatrix = np.zeros((numberOfClasses,numberOfClasses))
    errorCount = 0
    correctCount = 0
    for e in range(0,numberOfTestingExamples):
        #the prediction for each word is the index + 1 (to account of zero indexing) of the max value of the row  
        prediction = np.argmax(classifyProbabilityMatrix[e,:]) + 1
        realLabel = docIdToTestLabel[str(e + 1)]
        confusionMatrix[realLabel-1,prediction-1] = confusionMatrix[realLabel-1,prediction-1] + 1
        if prediction != realLabel:
            errorCount = errorCount + 1
        else:
            correctCount = correctCount + 1
    print("")
    print("Beta: ",beta)
    print("Accuracy: ",correctCount/numberOfTestingExamples)
    print(confusionMatrix)
    return correctCount/numberOfTestingExamples

#some constants to be used by the classify function
sizeOfVocabulary = 61188
numberOfClasses = 20
numberOfTestingSamples = 7505
numberOfTrainingSamples = 11269

#call the classify function for various values of beta
accuracy = classify(1.0/sizeOfVocabulary, sizeOfVocabulary,numberOfClasses,numberOfTestingSamples,numberOfTrainingSamples,False)
accuracyBeta0_00001 = classify(0.00001, sizeOfVocabulary,numberOfClasses,numberOfTestingSamples,numberOfTrainingSamples,False)
accuracyBeta0_0001 = classify(0.0001, sizeOfVocabulary,numberOfClasses,numberOfTestingSamples,numberOfTrainingSamples,False)
accuracyBeta0_001 = classify(0.001, sizeOfVocabulary,numberOfClasses,numberOfTestingSamples,numberOfTrainingSamples,False)
accuracyBeta0_01 = classify(0.01, sizeOfVocabulary,numberOfClasses,numberOfTestingSamples,numberOfTrainingSamples,False)
accuracyBeta0_1 = classify(0.1, sizeOfVocabulary,numberOfClasses,numberOfTestingSamples,numberOfTrainingSamples,False)
accuracyBeta1_0 = classify(1.0, sizeOfVocabulary,numberOfClasses,numberOfTestingSamples,numberOfTrainingSamples,False)

classify(1.0/sizeOfVocabulary, sizeOfVocabulary,numberOfClasses,numberOfTestingSamples,numberOfTrainingSamples,True)

#Uncomment to see plot, also uncomment the import above
'''
#Plot Accuracy vs. Beta
plt.plot([0.00001,0.0001,0.001,0.01,0.1,1.0],[accuracyBeta0_00001,accuracyBeta0_0001,accuracyBeta0_001,accuracyBeta0_01,accuracyBeta0_1,accuracyBeta1_0],'ro')
plt.margins(0.2, 0.2)
plt.xscale('log')
plt.ylabel('Accuracy')
plt.xlabel('Beta')
plt.show()
'''

