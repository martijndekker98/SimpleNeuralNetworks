import numpy as np

##
## Functions to compute features for the mnist dataset
##

def colourDistribution(image: np.ndarray):
    """ Get the colour distribution of a monochrome image (mnist dataset)
    :param image: the image that needs to be processed
    :return: a dict with the distribution of colours
    """
    dict_ = dict()
    for row in image:
        for pixel in row:
            if pixel in dict_:
                dict_[pixel] += 1
            else:
                dict_[pixel] = 1
    for key in dict_:
        writeLine(f"Key: {key} = {dict_[key]}")


def averageValue(image: np.ndarray):
    """ Get the average colour/intensity of a monochrome image (mnist dataset)
    :param image: the image that needs to be processed
    :return: the average colour/intensity value
    """
    ans = 0
    for row in image:
        for pixel in row:
            ans += pixel
    return ans / (image.shape[0] * image.shape[1])


def countMinVal(image: np.ndarray, minVal):
    """ Counts the number of pixels with a colour/intensirt >= minVal (mnist dataset)
    :param image: the image that needs to be processed
    :param minVal: the minimum colour/intesity value
    :return: the number of pixels for which the pixel value is >= minVal
    """
    c = 0
    for row in image:
        for pixel in row:
            if pixel >= minVal:
                c += 1
    return c


def findExtremesMinVal(image: np.ndarray, minVal):
    """ Get the left, right, top and bottom extremes in the image where the pixel values >= minVal
    IN OTHER WORDS: finds the bounding box for pixels, thus pixels outside this box will be guaranteed to have
    lower values than minVal
    :param image: the image that needs to be processed
    :param minVal: the minimum value for a pixel
    :return: the coordinates of the left, right, top and bottom extremes (bounding box coordinates)
    """
    l, r, t, b = findLeft(image, minVal), findRight(image, minVal), findTop(image, minVal), findBottom(image, minVal)
    return l, r, t, b


def findTop(image:np.ndarray, minVal):
    """ Get the top most coordinate (row) where at least 1 pixel >= minVal
    :param image: the image that needs to be processed
    :param minVal: the minimum value for a pixel
    :return: int that indicates the top row satisfying pixel >= minVal
    """
    for r, row in enumerate(image):
        for c, pixel in enumerate(row):
            if pixel >= minVal:
                return r

def findBottom(image: np.ndarray, minVal):
    """ Get the bottom most coordinate (row) where at least 1 pixel >= minVal
    :param image: the image that needs to be processed
    :param minVal: the minimum value for a pixel
    :return: int that indicates the bottom row satisfying pixel >= minVal
    """
    for i in range(1, image.shape[0]+1):
        for pixel in image[image.shape[0] - i]:
            if pixel >= minVal:
                return image.shape[0] - i

def findLeft(image: np.ndarray, minVal):
    """ Get the left most coordinate (column) where at least 1 pixel >= minVal
    :param image: the image that needs to be processed
    :param minVal: the minimum value for a pixel
    :return: int that indicates the left column satisfying pixel >= minVal
    """
    for i in range(0, image.shape[1]):
        for j in range(0, image.shape[0]):
            if image[j][i] >= minVal:
                return i

def findRight(image: np.ndarray, minVal):
    """ Get the right most coordinate (column) where at least 1 pixel >= minVal
    :param image: the image that needs to be processed
    :param minVal: the minimum value for a pixel
    :return: int that indicates the right column satisfying pixel >= minVal
    """
    for i in range(1, image.shape[1]+1):
        for j in range(1, image.shape[0]+1):
            if image[image.shape[0]-j][image.shape[1]-i] >= minVal:
                return image.shape[1]-i


def getXYList(image: np.ndarray, minVal):
    x, y = [], []
    writeLine(image.shape)
    for r, row in enumerate(image):
        for c, pixel in enumerate(row):
            if pixel >= minVal:
                x.append(c)
                y.append(r)
    return x, y


def pca(image: np.ndarray, minVal):
    """ Compute the PCA (principal component analysis) of the image
    :param image: the image that needs to be processed
    :param minVal: the minimum value for a pixel
    :return: the eigenvalues and eigenvectors, which indicate the spread of the pixels
    """
    x, y = getXYList(image, minVal)
    A_cov = np.cov(np.array([x, y]))
    writeLine(A_cov)
    eigenvalues, eigenvectors = np.linalg.eig(A_cov)
    writeLine(eigenvalues)
    writeLine(eigenvectors)
    return eigenvalues, eigenvectors


printing = False
def writeLine(txt: str):
    if printing: print(txt)

def computeFeature(image: np.ndarray, minVal):
    """ Compute the features for one monochrome image (mnist dataset)
    :param image: the image that needs to be processed
    :param minVal: the minimum value for a pixel
    :return: list: containing all the feature values for this image
    """
    l,r,t,b = findExtremesMinVal(image, minVal)
    height = b-t+1
    width = r-l+1
    eigenvalues, eigenvectors = pca(image, minVal)
    avg = averageValue(image)
    count = countMinVal(image, minVal)
    ans = [avg, count/(image.shape[0]*image.shape[1]), height/image.shape[0], width/image.shape[1],
            l/image.shape[1], r/image.shape[1], t/image.shape[0], b/image.shape[0]]
    ans.extend(eigenvalues)
    return ans

def computeFeatureList(images, minVal):
    """ Compute the feature values for each image in the dataset
    :param images: the images that need to be processed
    :param minVal: the minimum value for a pixel
    :return: list: with a list of feature values corresponding to one image in the dataset.
    """
    ans = [np.reshape(np.array(computeFeature(img, minVal)), (10,1)) for img in images]
    # ans = [np.array(computeFeature(img, minVal)) for img in images]
    return ans