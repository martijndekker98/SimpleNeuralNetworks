import tensorflow as tf
import matplotlib.pyplot as plt


def getData(count: int, norm: bool, reshaped: bool = False):
    """ Retrieve the mnist data
    :param count: the number of images in the train set
    :param norm: whether to normalise the data (pixel with RGB of (255, 128, 0) => (1.0, 0.5, 0.0))
    :param reshaped: whether to reshape the data for the CNN model
    :return: the train data+labels an the test data+labels
    """
    if reshaped:
        t, tl, te, tel = getDataPart(count, norm)
        return [a.reshape((1,28, 28)) for a in t], tl, [a.reshape((1,28,28)) for a in te], tel
    return getDataPart(count, norm)


def getDataPart(count: int, norm: bool):
    """ Retrieve the mnist data
    :param count: the number of images that should be taken from the mnist training data
    :param norm: whether to normalise the data (pixel with RGB of (255, 128, 0) => (1.0, 0.5, 0.0))
    :return: the train data+labels an the test data+labels
    """
    (trainImages, trainLabels), (testImages, testLabels) = tf.keras.datasets.mnist.load_data()
    if count == -1 and not norm:
        return trainImages, trainLabels, testImages, testLabels
    elif not norm:
        return trainImages[:count], trainLabels[:count], testImages[:count], testLabels[:count]
    trainImagesNorm, testImagesNorm = trainImages / 255.0, testImages / 255.0
    if count == -1 and norm:
        return trainImagesNorm, trainLabels, testImagesNorm, testLabels
    else:
        return trainImagesNorm[:count], trainLabels[:count], testImagesNorm[:count], testLabels[:count]


def getCifarData(count: int, norm: bool, reshaped: bool = False):
    """ Retrieve the data from the cifar10 dataset
    :param count: the number of images in the train set
    :param norm: whether to normalise the data (pixel with RGB of (255, 128, 0) => (1.0, 0.5, 0.0))
    :param reshaped: whether to reshape the data for the CNN model
    :return: the train data+labels an the test data+labels
    """
    if reshaped:
        t, tl, te, tel = getDataPartCifar(count, norm)
        return [a.reshape((1,28, 28)) for a in t], tl, [a.reshape((1,28,28)) for a in te], tel
    return getDataPartCifar(count, norm)


def getDataPartCifar(count: int, norm: bool):
    """ Retrieve the data from the cifar10 dataset
    :param count: the number of images in the train set
    :param norm: whether to normalise the data (pixel with RGB of (255, 128, 0) => (1.0, 0.5, 0.0))
    :return: the train data+labels an the test data+labels
    """
    (trainImages, trainLabels), (testImages, testLabels) = tf.keras.datasets.cifar10.load_data()
    if count == -1 and not norm:
        return trainImages, trainLabels, testImages, testLabels
    elif not norm:
        return trainImages[:count], trainLabels[:count], testImages[:count], testLabels[:count]
    trainImagesNorm, testImagesNorm = trainImages / 255.0, testImages / 255.0
    if count == -1 and norm:
        return trainImagesNorm, trainLabels, testImagesNorm, testLabels
    else:
        return trainImagesNorm[:count], trainLabels[:count], testImagesNorm[:count], testLabels[:count]


def showCifar10(train_images, train_labels):
    """ Shows 25 images from the cifar10 dataset
    :param train_images: images from the train set
    :param train_labels: corresponding labels
    :return: -
    """
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i])
        # The CIFAR labels happen to be arrays,
        # which is why you need the extra index
        plt.xlabel(class_names[train_labels[i][0]])
    plt.show()


def readResults(labels: list):
    """ Read the results of the various models trained/tested on cifar10
    :param labels: list of the names for the models (used to read .txt files)
    :return: the scores, losses and # of parameters for the models
    """
    scores, losses, parameters = [], [], []
    for label in labels:
        with open(f'cifar10/{label}.txt') as f:
            lines = f.readlines()
            scoresH, lossesH = [], []
            for score in lines[0].split('[')[1].split(']')[0].split(','):
                scoresH.append(float(score))
            for loss in lines[1].split('[')[1].split(']')[0].split(','):
                lossesH.append(float(loss))
            scores.append(scoresH)
            losses.append(lossesH)
            parameters.append(int(lines[2].split(':')[1]))
    return scores, losses, parameters