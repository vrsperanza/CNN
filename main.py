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

def fourierConv(img, filter):
	f = np.zeros(img.shape)	
	f[:filter.shape[0],:filter.shape[1]] = filter
	
	FTImg = np.fft.fft2(img)
	FTFilter = np.fft.fft2(f)
	
	return np.real(np.fft.ifft2(np.multiply(FTImg, FTFilter)))

def MLPXorTest():
	layer1 = DenseLayer(2, 5)
	layer2 = DenseLayer(5, 1)

	medianError = 0

	for train in range(100000):
		input = np.array((np.random.randint(0, 2), np.random.randint(0, 2)))
		output = input[0] ^ input[1]
		
		ans = layer2.forward(layer1.forward(input))
		
		medianError = (99*medianError+np.power(output-ans, 2))/100
		
		err = squaredErrorBackpropagation(output, ans)
		
		step = 0.1
		layer1.backpropagation(layer2.backpropagation(err, step), step)
		
		if train%1000 is 0:
			print(train, input, output, ans, (ans > 0.5) == (output > 0.5), medianError)
			
def ConvolutionTest():
	trainImages = mnist.trainingImages()
	
	kernelToApproach = np.matrix([[1/9,1/9,1/9],[1/9,1/9,1/9],[1/9,1/9,1/9]])
	
	
	step = 5e-5
	
	layer = ConvolutionLayer((3,3), 1)
	
	medianError = 1
	for i in range(100000):
		imgOrig = trainImages[random.randint(0, len(trainImages)-1)].astype(np.float32)/255;
		imgTarget = fourierConv(imgOrig, kernelToApproach);
		conv = layer.forward(imgOrig)
		err = squaredErrorBackpropagation(imgTarget, conv[0])
		layer.backpropagation(err, step)
		
		error = np.sqrt(np.mean(np.power(imgTarget-conv[0], 2)))
		medianError = (99*medianError+error)/100
		
		step *= 0.99995
		
		if i % 1000 is 0:
			print(layer.kernels[0])
			print(i, error, medianError, step)
	

	plt.subplot(3, 1, 1)
	plt.imshow(imgOrig, cmap='gray')

	plt.subplot(3, 1, 2)
	plt.imshow(imgTarget, cmap='gray')
	
	plt.subplot(3, 1, 3)
	plt.imshow(conv[0], cmap='gray')
	plt.show()

def MLPMNISTTest():
	trainImages = mnist.trainingImages()
	trainLabels = mnist.trainingLabels()
	
	denseLayer1 = DenseLayer(28*28, 100)
	denseLayer2 = DenseLayer(100, 3)
	
	medianError = 0

	for train in range(1000000):
		trainCase = random.randint(0, len(trainImages)-1)
		
		if(trainLabels[trainCase] >= 3):
			train -= 1
			continue
		
		input = trainImages[trainCase].ravel()
		output = oneHot(trainLabels[trainCase], 3)
		
		
		ans = denseLayer2.forward(denseLayer1.forward(input))
		
		medianError = (99*medianError+abs(output-ans))/100
		
		err = squaredErrorBackpropagation(output, ans)
		
		step = 0.01
		denseLayer1.backpropagation(denseLayer2.backpropagation(err, step), step)
		
		if train%100 is 0:
			print(train, output, ans, medianError.mean())