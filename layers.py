import numpy as np
def sigmoid(x):
	return 1/(1+np.exp(-x))

def d_sigmoid(x):
	return x*(1-x)

def squaredErrorBackpropagation(expectedResult, calculatedResult):
	return -2*(expectedResult-calculatedResult)
	
class DenseLayer:
	def __init__(self, inputSize, layerSize):
		self.weights = np.random.normal(loc=0, scale=1, size=(inputSize+1, layerSize))
		
	def forward(self, input):
		self.input = input
		self.output = sigmoid(np.matmul(np.append(input, [1], axis = 0), self.weights))
		return self.output

	def backpropagation(self, backpropagation, step):	
		backpropagation = backpropagation*d_sigmoid(self.output)
	
		inputAndTheta = np.append(self.input, [1], axis = 0)
		self.weights -= step*np.matmul(inputAndTheta.reshape(-1, 1), backpropagation.reshape(1, -1))

		weightSum = self.weights.sum(axis=1)
		return np.matmul(backpropagation.reshape(-1, 1), weightSum[:weightSum.shape[0]-1].reshape(1, -1)).sum(axis=0)
		
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
		
		self.input = inputMatrix
		self.output = []
		filter = np.zeros(matrix.shape)
		for kernel in self.kernels:
			filter[:kernel.shape[0],:kernel.shape[1]] = kernel
			kernelFFT = np.fft.fft2(filter)

			self.output.append( np.real(np.fft.ifft2(np.multiply(matrixFFT, kernelFFT))))#\
								#[int(kernel.shape[0]/2):int((matrix.shape[0]-kernel.shape[0]/2)), \
								#int(kernel.shape[1]/2):int((matrix.shape[1]-kernel.shape[1]/2))]))
		return self.output