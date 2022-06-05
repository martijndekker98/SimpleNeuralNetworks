import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import TFModel
import tensorflow as tf
import HillClimbing
import HelperFunctions as hf
import trainAndTest


# train and test a number of models on the mnist dataset
def trainAndRunMnist():
    trainAndTest.getResultsMNist()


# Read the results of the models on the cifar10 dataset and plot the results in a graph
def readAndPrintCifarResults():
    labels = ["Shallow fully connected", "Fully connected 256", "Simple CNN 16", "Simple CNN 32", "Simple CNN 32 dropout",
              "CNN double conv 16", "CNN double conv 32", "CNN double conv 32 dropout" ]
    testScores, losses, parameters = hf.readResults(labels)
    trainAndTest.plotResults(testScores, labels, parameters, losses)


# train and test a number of models on the Cifar10 dataset
def trainAndRunCifar():
    trainAndTest.getResultsCifar(-1, printResults=True)


# Use the hill climbing algorithm to find a good network for a feature model on the mnist data
def HillClimbingFeature():
    # Get the data
    trainFeatures, trainLabels, testFeatures, testLabels = hf.getFeatures(-1, 0.5)
    HillClimbing.HillClimbing(trainFeatures, trainLabels, testFeatures, testLabels, (10,1), 256, 5)


# Use the hill climbing algorithm to find a good network for a fully connected model on the mnist data
def HillClimbing():
    # Get the data
    trainImages, trainLabels, testImages, testLabels = hf.getData(-1, True, False)
    HillClimbing.HillClimbing(trainImages, trainLabels, testImages, testLabels, (28,28))


def main():
    # Read and print the result for the cifar10 dataset
    readAndPrintCifarResults()

    # Train and run on the cifar10 dataset
    # trainAndRunCifar()

    # Train and run on the mnist dataset
    # trainAndRunMnist()

    # Hill climbing for the feature model
    # HillClimbingFeature()

    # Hillclimbing
    # HillClimbing()


if __name__ == "__main__":
    main()