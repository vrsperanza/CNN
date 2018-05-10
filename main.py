import numpy as np
from layers import *
import mnist
import imageio
import matplotlib.pyplot as plt
import random

def oneHot(value, max):
	arr = np.zeros(max)
	arr[value] = 1
	return arr

def MNISTtrainCNN(cnn, step=0.000001, maxIterations=1000000, classes=3):
	trainImages = mnist.trainingImages()
	trainLabels = mnist.trainingLabels()
	
	medianError = 0

	for train in range(1000000):
		trainCase = random.randint(0, len(trainImages)-1)
	
		if(trainLabels[trainCase] >= classes):
			continue
		
		input = trainImages[trainCase]
		label = oneHot(trainLabels[trainCase], classes)
		
		ans = cnn.train(input, label, step)
	
		medianError = (99*medianError+abs(label-ans))/100
		
		if train%100 is 0:
			print(train, label, ans, medianError.mean())
	
def MLPMNISTTest(step = 0.01):
	cnn = CNN()
	cnn.addLayer(RavelLayer())
	cnn.addLayer(DenseLayer(28*28, 20))
	cnn.addLayer(SigmoidLayer())
	cnn.addLayer(DenseLayer(20, 3))
	cnn.addLayer(SigmoidLayer())
	
	MNISTtrainCNN(cnn)

def CNNMNISTTest(step = 0.01):
	cnn = CNN()
	
	cnn.addLayer(ConvolutionLayer((3,3), 4))
	cnn.addLayer(ReLULayer())
	cnn.addLayer(MaxPoolLayer((2,2), (2,2)))
	
	cnn.addLayer(RavelLayer())
	cnn.addLayer(DenseLayer(4*14*14, 10))
	cnn.addLayer(SigmoidLayer())
	cnn.addLayer(DenseLayer(10, 10))
	cnn.addLayer(SigmoidLayer())
	
	MNISTtrainCNN(cnn, classes=10)

CNNMNISTTest()