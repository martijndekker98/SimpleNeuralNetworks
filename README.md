# MNist
The [MNist](http://yann.lecun.com/exdb/mnist/) dataset consists of monochrome (black and white) images of 28 by 28 pixels of handwritten digits (0 to 9), with a training set of 60,000 images and a test set of 10,000 images. The goal for the models is to predict the correct digit (0 to 9) based on the image.

## Models
I have tested three different types of models for which there are two variants each. The first type of model is the feature model with the "Shallow features" and "Simple features" being the two variants. The second type is the fully connected neural network(FCNN) with a shallow and a simple version. Lastly, the convolutional neural network (CNN) has a simple variant and a double variant. <br>
Both the feature network and the FCNN have a shallow variant and a 'deep' variant. The shallow variant, as the name suggests, is shallow and only has an input and output layer, while the 'deep' variants have a hidden layer as well.
### Feature model
The feature model does not use the image itself to predict the label, but instead uses ten pre-computed features. These features then act as the input layer of a FCNN.
The ten features used are: average pixel value, percentage of pixels with a minimum value (0.5), the major eigenvalue, the minor eigenvalue and six features based on the bounding box, height, width, left, right, top, bottom. The reason that the features based on the bounding box are valid for this dataset is the normalisation of the digits done by the authors of the dataset (e.g., the digits were centered). The advantage of this feature model is the small size of the networks but it also comes with some major disadvantages. First of all, the performance is significantly lower than that of the FCNN and CNN. The second disadvantage is the computation time required for computing the feature values. <br>

### Fully connected neural network
The fully connected neural network(FCNN) first flattens the image (28x28) into an array of 784x1 which then acts as the input layer of the network. The model works the same as any other FCNN. One advantage of the FCNN is the higher performance than the feature model because every pixel can individually impact the next layer and thus the result. Moreover, the FCNN is better at taking the spatial information into account. One disadvantage is that the models grow quickly compared to the feature network and CNN, as adding a hidden layer will result in a major increase in the number of parameters.

### Convolutional neural network
The convolutional neural network (CNN) does not adjust the image using man-made processes in any way (no flattening or computing features) but instead uses two stages to process it. The first stage is the convolutional stage, where a 'filter' is moved over the image from left to right, top to bottom. This stage combines information from multiple pixels to reduce the number of neurons needed for the second stage. The second stage is the fully connected stage, which is essentially a small FCNN. The CNN has a number of advantages over the other model types, such as being spatial invariant, while it can take the spatial information into account. CNNs also tend to be smaller in terms of number of parameters, but this advantage is more pronounced with larger images.

## Results
<img src="figures/MNistResults.png" width="600" alt="results mnist"/>

# Cifar10
<img src="figures/Cifar10Results2.png" width="600" alt="results mnist"/>
