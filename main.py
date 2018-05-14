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

def oneHotEquals(a, b):
	for i in range(len(a)):
		if(abs(a[i]-b[i]) >= 0.5):
			return False
	return True
	
def MNISTtestCNN(cnn, classes=3):
	trainImages = mnist.testingImages()
	trainLabels = mnist.testingLabels()
	
	acc = 0
	tot = 0
	
	for i in range(len(trainImages)):
		if(trainLabels[i] >= classes):
			continue
			
		input = trainImages[i]
		label = oneHot(trainLabels[i], classes)
		
		tot += 1
		if oneHotEquals(cnn.estimate(input), label):
			acc += 1
			
	print(acc, tot, acc/tot)
	
def MNISTtrainCNN(cnn, step=0.000001, maxIterations=50000, classes=3):
	trainImages = mnist.trainingImages()
	trainLabels = mnist.trainingLabels()
	
	medianError = 0
	train = 0
	
	while train < maxIterations:
		trainCase = random.randint(0, len(trainImages)-1)
	
		if(trainLabels[trainCase] >= classes):
			continue
			
		train += 1
		
		input = trainImages[trainCase]
		label = oneHot(trainLabels[trainCase], classes)
		
		ans = cnn.train(input, label, step)
	
		medianError = (99*medianError+abs(label-ans))/100
		
		if train%1000 is 0:
			print(train, label, ans, medianError.mean())
	MNISTtestCNN(cnn, classes)

def MLPXorTest():
	cnn = CNN()
	cnn.addLayer(DenseLayer(2, 5))
	cnn.addLayer(SigmoidLayer())
	cnn.addLayer(DenseLayer(5, 1))
	cnn.addLayer(SigmoidLayer())

	medianError = 0

	for train in range(100000):
		input = np.array((np.random.randint(0, 2), np.random.randint(0, 2)))
		label = input[0] ^ input[1]
		
		ans = cnn.train(input, label, 0.01)
		
		medianError = (99*medianError+abs(label-ans))/100
		
		if train%1000 is 0:
			print(train, input, label, ans, (ans > 0.5) == (label > 0.5), medianError)
			
def MLPMNISTTest(step = 0.01, hiddenSize=10, classes=2, maxIterations=200000):
	cnn = CNN()
	cnn.addLayer(RavelLayer())
	cnn.addLayer(DenseLayer(28*28, hiddenSize))
	cnn.addLayer(SigmoidLayer())
	cnn.addLayer(DenseLayer(hiddenSize, classes))
	cnn.addLayer(SigmoidLayer())
	
	MNISTtrainCNN(cnn, step = step, classes=classes, maxIterations=maxIterations)

def CNNMNISTTest(step = 0.01):
	cnn = CNN()
	
	cnn.addLayer(ConvolutionLayer((3,3), 4))
	cnn.addLayer(ReLULayer())
	cnn.addLayer(MaxPoolLayer((2,2), (2,2)))
	
	cnn.addLayer(RavelLayer())
	cnn.addLayer(DenseLayer(4*14*14, 10))
	cnn.addLayer(SigmoidLayer())
	cnn.addLayer(DenseLayer(10, 2))
	cnn.addLayer(SigmoidLayer())
	
	MNISTtrainCNN(cnn, classes=2)

MLPMNISTTest()