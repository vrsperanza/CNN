import numpy as np
from layers import *
import imageio
import matplotlib.pyplot as plt

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
	# Le a entrada e carrega as imagens
	imgOrig = imageio.imread("moon.png")

	layer = ConvolutionLayer((3,3), 2)
	conv = layer.forward(imgOrig)

	plt.subplot(3, 1, 1)
	plt.imshow(imgOrig, cmap='gray')

	plt.subplot(3, 1, 2)
	plt.imshow(conv[0], cmap='gray')

	plt.subplot(3, 1, 3)
	plt.imshow(conv[1], cmap='gray')

	plt.show()

ConvolutionTest();