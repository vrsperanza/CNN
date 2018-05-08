import numpy as np
def sigmoid(x):
	return 1/(1+np.exp(-x))

def d_sigmoid(x):
	return x*(1-x)

def squaredErrorBackpropagation(expectedResult, calculatedResult):
	return -2*(expectedResult-calculatedResult)
	
class NeuronLayer:
	def __init__(self, inputSize, layerSize):
		self.weights = np.random.normal(loc=0, scale=1, size=(inputSize+1, layerSize))
		
	def forward(self, input):
		self.input = input
		self.forward_ = sigmoid(np.matmul(np.append(input, [1], axis = 0), self.weights))
		return self.forward_

	def backpropagation(self, backpropagation, step):	
		backpropagation = backpropagation*d_sigmoid(self.forward_)
	
		inputAndTheta = np.append(self.input, [1], axis = 0)
		self.weights -= step*np.matmul(inputAndTheta.reshape(-1, 1), backpropagation.reshape(1, -1))

		weightSum = self.weights.sum(axis=1)
		return np.matmul(backpropagation.reshape(-1, 1), weightSum[:weightSum.shape[0]-1].reshape(1, -1)).sum(axis=0)
		
layer1 = NeuronLayer(2, 5)
layer2 = NeuronLayer(5, 1)

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
