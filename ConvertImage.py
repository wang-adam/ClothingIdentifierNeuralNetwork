import numpy as np
import cv2
import matplotlib.pyplot as plt
import foregrounddetector as fd


# Compresses the given image down to a 28x28 grayscale image, which the Neural Network uses.
def compress(fileName):
    image, oldimage = fd.remove_background(fileName)
    print(image.shape)
    data = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)/255
    # test = np.arange(0, 100, 1).reshape((10, 10))
    # data = data[200:-200, 200:-200]
    arraySize = 28
    row = int(np.floor(data.shape[0]/arraySize))
    col = int(np.floor(data[0].shape[0]/arraySize))
    newData = np.zeros((1, arraySize, arraySize))
    for i in range(arraySize):
        for j in range(arraySize):
            newData[0, i, j] = np.sum(
                data[i*row:(i+1)*row, j*col:(j+1)*col])
    # plt.subplot(1, 2, 2)
    # plt.imshow(newData[0])
    # plt.colorbar()
    return newData, oldimage


def compress1(fileName):
    image = cv2.imread(fileName)
    print(image.shape)
    data = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)/255
    # test = np.arange(0, 100, 1).reshape((10, 10))
    # data = data[200:-200, 200:-200]
    arraySize = 28
    row = int(np.floor(data.shape[0]/arraySize))
    col = int(np.floor(data[0].shape[0]/arraySize))
    newData = np.zeros((1, arraySize, arraySize))
    for i in range(arraySize):
        for j in range(arraySize):
            newData[0, i, j] = np.sum(
                data[i*row:(i+1)*row, j*col:(j+1)*col])
    # plt.subplot(1, 2, 1)
    # plt.imshow(1-newData[0])
    # plt.colorbar()
    # plt.show()
    return 1-newData, image


# oldData, oldimage = compress2("coat.jpg")
# newData, newimage = compress("coat.jpg")

# np.savetxt("senddata.txt", oldData[0]-newData[0])
# np.savetxt("image.txt", oldimage[0]-newimage[0])

# plt.show()
