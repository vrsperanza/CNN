import numpy as np

def squaredErrorBackpropagation(expectedResult, calculatedResult):
	return -2*(expectedResult-calculatedResult)
	
class Sigmoid:
	def forward(self, input):
		self.output = 1/(1+np.exp(-input))
		return self.output
		
	def backward(self, backpropagation, step):
		return backpropagation*self.output*(1-self.output)

class ReLU:
	def forward(self, input):
		self.output = np.maximum(np.zeros_like(input), input)
		return self.output
		
	def backward(self, backpropagation, step):
		self.output[self.output > 0] = 1
		return self.output
	
		
class DenseLayer:
	def __init__(self, inputSize, layerSize):
		self.weights = np.random.normal(loc=0, scale=1, size=(inputSize+1, layerSize))
		
	def forward(self, input):
		self.input = input
		self.output = np.matmul(np.append(input, [1], axis = 0), self.weights)
		return self.output
		
	def backward(self, backpropagation, step):	
		inputAndTheta = np.append(self.input, [1], axis = 0)
		self.weights -= step*np.matmul(inputAndTheta.reshape(-1, 1), backpropagation.reshape(1, -1))

		weightMean = self.weights.mean(axis=1)
		return np.matmul(backpropagation.reshape(-1, 1), weightMean[:weightMean.shape[0]-1].reshape(1, -1)).sum(axis=0)
		
class ConvolutionLayer:
	def __init__(self, kernelShape, kernelAmount):
		self.kernelShape = kernelShape
		self.kernelAmount = kernelAmount
		self.kernels = []
		for i in range(kernelAmount):
			self.kernels.append(np.random.normal(loc=0, scale=1, size=(kernelShape[0], kernelShape[1])))
			print(self.kernels[i])
	
	def forward(self, inputMatrix):
		matrix = np.pad(inputMatrix, (int(self.kernelShape[0]/2), int(self.kernelShape[1]/2)), 'constant')
		matrixFFT = np.fft.fft2(matrix)
		
		self.input = matrix
		self.output = []
		filter = np.zeros(matrix.shape)
		for kernel in self.kernels:
			filter[:kernel.shape[0],:kernel.shape[1]] = kernel
			kernelFFT = np.fft.fft2(filter)

			self.output.append( np.real(np.fft.ifft2(np.multiply(matrixFFT, kernelFFT)) \
								[int(kernel.shape[0]/2):int((matrix.shape[0]-kernel.shape[0]/2)+1), 
								int(kernel.shape[1]/2):int((matrix.shape[1]-kernel.shape[1]/2)+1)]))
		return self.output
		
	def backward(self, backpropagation, step):
		for k in range(len(backpropagation)):
			kernel = self.kernels[k]
			for i in range(kernel.shape[0]):
				for j in range(kernel.shape[1]):
					kernel[i,j] -= step*np.mean(np.dot(self.input[i:i+backpropagation[k].shape[0],j:j+backpropagation[k].shape[1]], backpropagation[k]))
		
		# TO-DO: optionally propagate error

class MaxPoolLayer:
	def __init__(self, poolShape, poolStride):
		self.poolShape = poolShape
		self.poolStride = poolStride
		
class CNN:
	def __init__(self):
		self.layers = []
		
	def addLayer(self, layer):
		self.layers.append(layer)
		
	def estimate(self, input):
		output = input
		for layer in self.layers:
			output = layer.forward(output)
		return output
	
	def train(self, input, label, step):
		output = self.estimate(input)
		backpropagation = squaredErrorBackpropagation(label, output)
		for layer in reversed(self.layers):
			backpropagation = layer.backward(backpropagation, step)
		
		return output