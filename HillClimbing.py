import tensorflow as tf
from keras.layers import Conv2D, Flatten, Dense, MaxPooling2D
import TFModel
from sklearn.model_selection import train_test_split

##
## Simple hillclimbing used for finding the right layersize for some of the simpler models
##

class ClimbingStep():
    def __init__(self, options, epochs, result, model):
        self.options    = options
        self.epochs     = epochs
        self.valScore   = result
        self.model      = model

    def print(self):
        print(f"Model obtained {self.valScore} with {self.options} neurons ({self.epochs} epochs)")

    def toString(self):
        return f"Model obtained {self.valScore} with {self.options} neurons ({self.epochs} epochs)"

    def getOptions(self, includeEpochs: bool = True):
        return (self.options, self.epochs) if includeEpochs else self.options


class Variables(object):
    def __init__(self, options: list):
        self.options = options

    def __getitem__(self, item):
        return self.options[item]

    def __eq__(self, other):
        if not isinstance(other, Variables): return False
        if len(self.options) != len(other.options): return False
        for i in len(self.options):
            if self.options[i] != other.options[i]: return False
        return True



class ClimbingStepExtended():
    def __init__(self, layerSize, epochs, result, model):
        self.layerSize  = layerSize
        self.epochs     = epochs
        self.valScore   = result
        self.model      = model

    def print(self):
        print(f"Model obtained {self.valScore} with {self.layerSize} neurons ({self.epochs} epochs)")

    def toString(self):
        return f"Model obtained {self.valScore} with {self.layerSize} neurons ({self.epochs} epochs)"

    def getOptions(self):
        return (self.layerSize, self.epochs)



def HillClimbing(tr, trL, test, testL, inputSize, startSize: int = 128, startEpochs: int = 5):
    print("Start the hill climbing")
    train, val, trainL, valL = train_test_split(tr, trL, test_size=0.2)

    startModel = makeAndEvaluateModel(train, trainL, val, valL, inputSize, (startSize, startEpochs))
    print(f"Start model: {startModel.toString()}")

    h = startModel
    visited = {}
    visited[h.getOptions()] = h
    improving = True
    while improving:
        print(f"Currently the best model has val score: {h.valScore}")
        neighbs = generateNeighboursOld(h.getOptions())

        nModels = []
        for n in neighbs:
            if not n in visited:
                nModels.append(makeAndEvaluateModel(train, trainL, val, valL, inputSize, n))
        highest = 0
        for index, nmodel in enumerate(nModels):
            if nmodel.valScore > nModels[highest].valScore:
                highest = index

        if nModels[highest].valScore > h.valScore:
            h = nModels[highest]
        else:
            improving = False
    print(f"Found top of hill with variables {h.getOptions()} val score: {h.valScore}")
    print(f"Score on the test set: {evaluate(h.model, test, testL)[1]}")


#def generateNeighbours(options: list):


def generateNeighboursOld(options: list, addSubtract: bool = True, layerAddition: int = 4, multiplyDivide: bool = True,
                          changeEpochs: bool = True, epochsAddition: int = 1):
    ans = []
    if addSubtract: ans.extend([(options[0] + layerAddition, options[1]), (options[0] - layerAddition, options[1])])
    if multiplyDivide: ans.extend([(int(options[0] * 0.5), options[1]), (int(options[0] * 2), options[1])])
    if changeEpochs: ans.extend([(options[0], options[1] + epochsAddition), (options[0], options[1] - epochsAddition)])
    return ans


def makeAndEvaluateModel(train, trainL, test, testL, inputSize, options: list):
    if not isinstance(options[0], int) or not isinstance(options[1], int):
        print(f"Error, options should contain integers: {options}")
        return
    model = TFModel.FullyConnected(inputSize, options[0])
    # model = TFModel.SimpleFeatureModel(inputSize, options[0])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(train, trainL, epochs=options[1])
    results = evaluate(model, test, testL)
    return ClimbingStep(options[0], options[1], results[1], model)


def evaluate(model, test, testL):
    results = model.evaluate(test, testL)
    return results