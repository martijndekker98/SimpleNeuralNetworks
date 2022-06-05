from sklearn.model_selection import train_test_split

import TFModel as tfm
import tensorflow as tf
import TFModel
import HelperFunctions as hf
import Features
import numpy as np
import matplotlib.pyplot as plt
from Features import pca, computeFeature, computeFeatureList


def getFeatures(count: int, minVal):
    """ Gets the features for the images in the train and test set (mnist)
    :param count: the number of images in the train set
    :param minVal: the minimum value, used for computing the features
    :return: array: features for the train images, train labels, features for the test images, test labels
    """
    tr, trl, te, tel = hf.getData(count, True)
    trainFeatures, testFeatures = computeFeatureList(tr, minVal), computeFeatureList(te, minVal)
    return np.array(trainFeatures), trl, np.array(testFeatures), tel


def compileTrainAndEvaluate(model, epochs: int, train, trainL, test, testL, earlyStopping: bool = False):
    """ Compile, train and test a model
    :param model: the ML network used for training/testing
    :param epochs: number of epochs to train for
    :param train: the training data
    :param trainL: the training labels
    :param test: the testing data
    :param testL: the testing labels
    :param earlyStopping: bool: whether to use earlystopping or not
    :return: the accuracy on the test set and the number of trainable parameters in the model
    """
    if earlyStopping: return compileTrainAndEvaluateES(model, train, trainL, test, testL)
    model.compileModel()
    model.fit(train, trainL, epochs=epochs)
    results = model.evaluate(test, testL)
    trainParams = np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    return results[1], trainParams


def compileTrainAndEvaluateES(model, train1, trainL1, test, testL):
    """ Compile, train and test a model using early stopping
    :param model: the ML network used for training/testing
    :param train1: the training data
    :param trainL1: the training labels
    :param test: the testing data
    :param testL: the testing labels
    :return: the accuracy on the test set and the number of trainable parameters in the model
    """
    earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
    train, val, trainL, valL = train_test_split(train1, trainL1, test_size=0.2)

    model.compileModel()
    model.fit(train, trainL, epochs=100, validation_data=(val, valL), callbacks=[earlyStopping])
    results = model.evaluate(test, testL)
    trainParams = np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    return results[1], trainParams


def getResultsMNist(count: int = -1):
    """ Trains a number of models on the mnist data, then tests them and will plot the results to show the differences
    between the various models used.
    :param count: the number of images to use (from the train set)
    :return: -
    """
    epochs = [1, 2, 5, 10]
    trainFeatures, trainLabels, testFeatures, testLabels = getFeatures(count, 0.5)
    shallowFeat, simpleFeat = [], []
    for epoch in epochs:
        model = TFModel.ShallowFeatureModel((10, 1))
        results, params = compileTrainAndEvaluate(model, epoch, trainFeatures, trainLabels, testFeatures, testLabels)
        shallowFeat.append([results, params])
    for epoch in epochs:
        model = TFModel.SimpleFeatureModel((10, 1), 256)
        results, params = compileTrainAndEvaluate(model, epoch, trainFeatures, trainLabels, testFeatures, testLabels)
        simpleFeat.append([results, params])

    train, trainL, test, testL = hf.getData(count, True, False)
    shallowFully, simpleFully = [], []
    for epoch in epochs:
        model = TFModel.ShallowFullyConnected((28,28))
        results, params = compileTrainAndEvaluate(model, epoch, train, trainL, test, testL)
        shallowFully.append([results, params])
    for epoch in epochs:
        model = TFModel.FullyConnected((28,28), 256)
        results, params = compileTrainAndEvaluate(model, epoch, train, trainL, test, testL)
        simpleFully.append([results, params])

    trainCNN, trainLCNN, testCNN, testLCNN = hf.getData(count, True, True)
    trainCNN = np.array([a.reshape((28, 28, 1)) for a in trainCNN])
    testCNN = np.array([a.reshape((28, 28, 1)) for a in testCNN])
    cnn1, cnn2 = [], []
    for epoch in epochs:
        model = TFModel.SimpleCNN((28,28,1))
        results, params = compileTrainAndEvaluate(model, epoch, trainCNN, trainLCNN, testCNN, testLCNN)
        cnn1.append([results, params])
    for epoch in epochs:
        model = TFModel.SimpleCNN2((28,28,1))
        results, params = compileTrainAndEvaluate(model, epoch, trainCNN, trainLCNN, testCNN, testLCNN)
        cnn2.append([results, params])
    xsteps = [1,2,3,4]
    # plt.plot(xsteps, [a[0] for a in shallowFeat], "-bs", label=f"Shallow features ({shallowFeat[0][1]})")
    # plt.plot(xsteps, [a[0] for a in simpleFeat], "-rs", label=f"Simple features ({simpleFeat[0][1]})")
    # plt.plot(xsteps, [a[0] for a in shallowFully], "-b^", label=f"Shallow fully connected ({shallowFully[0][1]})")
    # plt.plot(xsteps, [a[0] for a in simpleFully], "-r^", label=f"Simple fully connected ({simpleFully[0][1]})")
    # plt.plot(xsteps, [a[0] for a in cnn1], "-bo", label=f"Simple CNN ({cnn1[0][1]})")
    # plt.plot(xsteps, [a[0] for a in cnn2], "-ro", label=f"Double CNN ({cnn2[0][1]})")
    plt.plot(xsteps, [a[0] for a in shallowFeat], "-b^", label=f"Shallow features ({shallowFeat[0][1]})")
    plt.plot(xsteps, [a[0] for a in simpleFeat], "-bs", label=f"Simple features ({simpleFeat[0][1]})")
    plt.plot(xsteps, [a[0] for a in shallowFully], "-r^", label=f"Shallow fully connected ({shallowFully[0][1]})")
    plt.plot(xsteps, [a[0] for a in simpleFully], "-rs", label=f"Simple fully connected ({simpleFully[0][1]})")
    plt.plot(xsteps, [a[0] for a in cnn1], "-g^", label=f"Simple CNN ({cnn1[0][1]})")
    plt.plot(xsteps, [a[0] for a in cnn2], "-gs", label=f"Double Ch. CNN ({cnn2[0][1]})")
    plt.xlabel('# of epochs')
    plt.ylabel('Test accuracy')
    plt.yticks(np.arange(0.1, 1.1, 0.1))
    plt.xticks([1,2,3,4], [1,2,5,10])
    plt.legend(loc="lower right")
    plt.title("Test accuracies MNIST")

    # print stuff
    lijsten = [shallowFeat, simpleFeat, shallowFully, simpleFully, cnn1, cnn2]
    for lijst in lijsten:
        print(f"Vals: {[a[0] for a in lijst]} ({lijst[0][1]})")

    plt.show()


def getResultsCifar(count: int = -1, printResults: bool = True):
    """ Trains a number of models on the cifar10 data, then tests them and (possibly) plot the results to show the differences
    between the various models used.
    :param count: the number of images to use (from the train set)
    :param printResults: bool: whether to print and plot the results
    :return: -
    """
    epochs = [1, 4, 5, 5, 10, 10, 15]  # 1, 5, 10, 15, 25, 35, 50
    trainC, trainLC, testC, testLC = hf.getCifarData(count, True)

    modelSFC = TFModel.ShallowFullyConnected((32, 32, 3))
    modelFC = TFModel.FullyConnected((32, 32, 3), 256)
    modelCNN1_16 = TFModel.CNN1((32,32,3), 16)
    modelCNN1_32 = TFModel.CNN1((32,32,3), 32)
    modelCNN1_dropout = TFModel.CNN1((32,32,3), 32, 0.2)
    modelCNN1_DC_16 = TFModel.CNN1DoubleConv((32,32,3), 16)
    modelCNN1_DC_32 = TFModel.CNN1DoubleConv((32,32,3), 32)
    modelCNN1_DC_dropout = TFModel.CNN1DoubleConv((32,32,3), 32, 0.2)
    modellen = [modelSFC, modelFC, modelCNN1_16, modelCNN1_32, modelCNN1_dropout, modelCNN1_DC_16, modelCNN1_DC_32, modelCNN1_DC_dropout]
    labels = ["Shallow fully connected", "Fully connected 256", "Simple CNN 16", "Simple CNN 32", "Simple CNN 32 dropout",
              "CNN double conv 16", "CNN double conv 32", "CNN double conv 32 dropout" ]

    parameters, testScores, losses = [], [], []
    for c, model in enumerate(modellen):
        if c != 4: continue
        model.compileModel()
        ans, lossesM = [], []
        for epochsToTrain in epochs:
            model.fit(trainC, trainLC, epochs=epochsToTrain)
            results = model.evaluate(testC, testLC)
            ans.append(results[1])
            lossesM.append(results[0])
        trainParams = np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
        parameters.append(trainParams)
        # print(f"Model: {model} with {trainParams} parameters")
        testScores.append(ans)
        losses.append(lossesM)
        if printResults: writeToFile(ans, lossesM, f"{labels[c]}.txt", trainParams)
    print(parameters)
    if printResults: plotResults(testScores, labels, parameters, losses)


def plotResults(testScores: list, labels: list, parameters: list, losses: list):
    """
    :param testScores: the test accuracies for the models
    :param labels: the labels of the models, e.g. "Shallow fully connected"
    :param parameters: the number of parameters per model
    :param losses: list with the test losses per model
    :return: -
    """
    colours = ["-b^", "-bs", "-r^", "-rs", "-ro", "-g^", "-gs", "-go", "-m^", "-ms", "-mo"]
    xsteps = [1, 2, 3, 4, 5, 6, 7]
    for index, label in enumerate(labels):
        plt.plot(xsteps, testScores[index], colours[index], label=f"{label} ({parameters[index]})")
    plt.xlabel('# of epochs')
    plt.ylabel('Test accuracy')
    plt.yticks(np.arange(0.1, 0.9, 0.1))
    plt.xticks([1, 2, 3, 4, 5, 6, 7], [1, 5, 10, 15, 25, 35, 50])
    plt.legend(loc="lower right")
    plt.title("Test accuracies Cifar10")

    # print stuff
    for ind, lijst in enumerate(testScores):
        print(f"Vals: {lijst} ({labels[ind]})")
    for ind, lijst in enumerate(losses):
        print(f"Losses: {lijst} ({labels[ind]})")

    plt.show()


def writeToFile(vals, losses, fileName, parameters):
    """ Writes the results of testing to a txt file
    :param vals: a list with the accuracties for a model
    :param losses: a list with the losses for a model
    :param fileName: the name to use for writing the file
    :param parameters: the number of parameters
    :return: -
    """
    f = open(f"cifar10/{fileName}", "w")
    f.write(f"Vals: {vals}\n")
    f.write(f"Losses: {losses}\n")
    f.write(f"Parameters: {parameters}\n")