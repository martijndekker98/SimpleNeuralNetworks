import tensorflow as tf
from keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, Dropout


class TFModel1(tf.keras.Model):
    def __init__(self):
        super(TFModel1, self).__init__()
        self.conv1 = Conv2D()


class ShallowFeatureModel(tf.keras.Model):
    def __init__(self, inp_shape):
        super(ShallowFeatureModel, self).__init__()
        self.flat = Flatten(input_shape=inp_shape)
        self.outp = Dense(10, activation='softmax')

    def call(self, x):
        x = self.flat(x)
        return self.outp(x)

    def compileModel(self):
        self.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


class SimpleFeatureModel(tf.keras.Model):
    def __init__(self, inp_shape, layerSize: int):
        super(SimpleFeatureModel, self).__init__()
        self.flat   = Flatten(input_shape=inp_shape)
        self.D1     = Dense(layerSize, activation='relu')
        self.outp   = Dense(10, activation='softmax')

    def call(self, x):
        x = self.flat(x)
        x = self.D1(x)
        return self.outp(x)

    def compileModel(self):
        self.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])



class FullyConnectedSimple(tf.keras.Model):
    def __init__(self, inp_shape):
        super(FullyConnectedSimple, self).__init__()
        self.flat   = Flatten(input_shape=inp_shape)
        self.D1     = Dense(196, activation='relu')
        self.outp   = Dense(10, activation='softmax')

    def call(self, x):
        x = self.flat(x)
        x = self.D1(x)
        return self.outp(x)

    def compileModel(self):
        self.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


class FullyConnected(tf.keras.Model):
    def __init__(self, inp_shape, layerSize: int):
        super(FullyConnected, self).__init__()
        self.flat   = Flatten(input_shape=inp_shape)
        self.D1     = Dense(layerSize, activation='relu')
        self.outp   = Dense(10, activation='softmax')

    def call(self, x):
        x = self.flat(x)
        x = self.D1(x)
        return self.outp(x)

    def compileModel(self):
        self.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


class ShallowFullyConnected(tf.keras.Model):
    def __init__(self, inp_shape):
        super(ShallowFullyConnected, self).__init__()
        self.flat   = Flatten(input_shape=inp_shape)
        self.outp   = Dense(10, activation='softmax')

    def call(self, x):
        x = self.flat(x)
        return self.outp(x)

    def compileModel(self):
        self.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


class SimpleCNN(tf.keras.Model):
    def __init__(self, inp_shape):
        super(SimpleCNN, self).__init__()
        self.conv1  = Conv2D(8, (3,3), activation='relu', input_shape=inp_shape)
        self.pool1  = MaxPooling2D((2,2))
        self.conv2  = Conv2D(16, (2,2), activation='relu')
        self.pool2  = MaxPooling2D((2,2))

        # Flatten
        self.flat   = Flatten()
        self.outp   = Dense(10, activation='softmax')

    def call(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)

        x = self.flat(x)
        return self.outp(x)

    def compileModel(self):
        self.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

class SimpleCNN2(tf.keras.Model):
    def __init__(self, inp_shape):
        super(SimpleCNN2, self).__init__()
        self.conv1  = Conv2D(16, (3,3), activation='relu', input_shape=inp_shape)
        self.pool1  = MaxPooling2D((2,2))
        self.conv2  = Conv2D(32, (2,2), activation='relu')
        self.pool2  = MaxPooling2D((2,2))

        # Flatten
        self.flat   = Flatten()
        self.outp   = Dense(10, activation='softmax')

    def call(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)

        x = self.flat(x)
        return self.outp(x)

    def compileModel(self):
        self.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])








class CNN1(tf.keras.Model):
    def __init__(self, inp_shape, channelCount: int):
        super(CNN1, self).__init__()
        self.conv1 = Conv2D(channelCount, (3, 3), activation='relu', input_shape=inp_shape)
        self.pool1 = MaxPooling2D((2, 2))
        self.conv2 = Conv2D(channelCount*2, (2, 2), activation='relu')
        self.pool2 = MaxPooling2D((2, 2))
        self.conv3 = Conv2D(channelCount*4, (2, 2), activation='relu')
        self.pool3 = MaxPooling2D((2, 2))

        # Flatten
        self.flat = Flatten()
        self.outp = Dense(10, activation='softmax')

    def call(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)

        x = self.flat(x)
        return self.outp(x)

    def summary(self):
        x = tf.keras.Input(shape=(32, 32, 3))
        model = tf.keras.Model(inputs=[x], outputs=self.call(x))
        return model.summary()

    def compileModel(self):
        self.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


class CNN1DoubleConv(tf.keras.Model):
    def __init__(self, inp_shape, channelCount: int, dropout: float = 0.0):
        super(CNN1DoubleConv, self).__init__()
        self.dropoutVal = dropout
        self.conv1 = Conv2D(channelCount, (3, 3), activation='relu', input_shape=inp_shape)
        self.conv2 = Conv2D(channelCount, (3, 3), activation='relu')
        self.pool1 = MaxPooling2D((2, 2))
        self.drop1 = Dropout(dropout)
        self.conv3 = Conv2D(channelCount * 2, (3, 3), activation='relu')
        self.conv4 = Conv2D(channelCount * 2, (3, 3), activation='relu')
        self.pool2 = MaxPooling2D((2, 2))
        self.drop2 = Dropout(dropout)

        # Flatten
        self.flat = Flatten()
        self.fc1 = Dense(128, activation='relu')
        self.drop3 = Dropout(dropout)
        self.outp = Dense(10, activation='softmax')

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        if self.dropoutVal > 0.0: x = self.drop1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        if self.dropoutVal > 0.0: x = self.drop2(x)

        x = self.flat(x)
        x = self.fc1(x)
        if self.dropoutVal > 0.0: x = self.drop3(x)
        return self.outp(x)

    def summary(self):
        x = tf.keras.Input(shape=(32, 32, 3))
        model = tf.keras.Model(inputs=[x], outputs=self.call(x))
        return model.summary()

    def compileModel(self):
        self.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# https://machinelearningmastery.com/how-to-develop-a-cnn-from-scratch-for-cifar-10-photo-classification/
class CNNVGG3(tf.keras.Model):
    def __init__(self, inp_shape, channelCount: int, dropout: float = 0.0):
        super(CNNVGG3, self).__init__()
        self.dropoutVal = dropout
        self.conv1 = Conv2D(channelCount, (3, 3), activation='relu', input_shape=inp_shape)
        self.conv2 = Conv2D(channelCount, (3, 3), activation='relu')
        self.pool1 = MaxPooling2D((2, 2))
        self.drop1 = Dropout(dropout)
        self.conv3 = Conv2D(channelCount * 2, (3, 3), activation='relu')
        self.conv4 = Conv2D(channelCount * 2, (3, 3), activation='relu')
        self.pool2 = MaxPooling2D((2, 2))
        self.drop2 = Dropout(dropout)
        self.conv5 = Conv2D(channelCount * 4, (3, 3), activation='relu')
        self.conv6 = Conv2D(channelCount * 4, (3, 3), activation='relu')
        self.drop3 = Dropout(dropout)

        # Flatten
        self.flat = Flatten()
        self.fc1 = Dense(128, activation='relu')
        self.drop4 = Dropout(dropout)
        self.outp = Dense(10, activation='softmax')

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        if self.dropoutVal > 0.0: x = self.drop1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        if self.dropoutVal > 0.0: x = self.drop2(x)
        x = self.conv5(x)
        x = self.conv6(x)
        if self.dropoutVal > 0.0: x = self.drop3(x)

        x = self.flat(x)
        x = self.fc1(x)
        if self.dropoutVal > 0.0: x = self.drop4(x)
        return self.outp(x)

    def summary(self):
        x = tf.keras.Input(shape=(32, 32, 3))
        model = tf.keras.Model(inputs=[x], outputs=self.call(x))
        return model.summary()

    def compileModel(self):
        self.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])