
import numpy as np
import math




class mlp:
	np.seterr(over='ignore')
	np.seterr(invalid='ignore')


	def __init__(self, inputs, targets, nhidden, beta = 1,eta=0.1, bias =-1):
		self.beta = beta #sigmoed function parameter
		self.eta = eta #learning rate
		self.bias= bias #bias
		self.momentum = 0.0
		self.n_illa = inputs.shape[0]
		assert self.n_illa ==targets.shape[0]
		self.ninputs = inputs.shape[1]+1
		self.nhidden = nhidden+1
		self.outputs= targets.shape[1]
		#set random weights.
		self.hidden_l_weights= 2*np.random.random((self.ninputs, self.nhidden))-1
		self.output_l_weights = 2*np.random.random((self.nhidden, self.outputs))-1



		#print('To be implemented')

	def earlystopping(self, inputs, targets, valid, validtargets):
		#bias on the inputs
		inputb = np.ones((inputs.shape[0], 1))*self.bias
		inputb2 = np.concatenate((inputb, inputs), axis=1)

		
		#bias on the valid
		validb = np.ones((valid.shape[0], 1))*self.bias
		valid2 = np.concatenate((validb, valid), axis=1)

		Verror = float('inf')
		m_err=0.01
		it = 0
		#computes error as fucnction of outputs and targets .

		while  Verror > m_err:
			self.train(inputb2, targets)
			hiddenL, outputR =  self.forward(valid2) # going forward to get activation of hidden and output layers
			updatedError = np.sum((outputR-validtargets)**2)/self.n_illa

			if updatedError<Verror:
				Verror=updatedError
			else:
				break
			it +=1	
	

	def train(self, inputs, targets, iterations=100):

		for i in range(iterations):
			hiddenL, outputL = self.forward(inputs) # going forward to get activation of hidden and output layers
			
			#going backwards, sending error back through the network. Backpropagation
			outputdelta = (outputL - targets)*(outputL*(1-outputL))						 #sigmoid derivate function
			hiddendelta = outputdelta.dot(self.output_l_weights.T)*(hiddenL*(1-hiddenL)) #sigmod derivate function
			
			#update weights from the computed deltas
			self.output_l_weights -= self.eta*hiddenL.T.dot(outputdelta)
			self.hidden_l_weights -= self.eta*inputs.T.dot(hiddendelta)

		
		

	def forward(self, inputs):
		#run forward, use weights and inputs to calculate activation of hidden and use those 
		#to calculate for output layer

		hiddenL = 1/(1+np.exp(-self.beta*(np.dot(inputs, self.hidden_l_weights)))) #activation with sigmoid function
		hiddenL[:, 0] = self.bias
		outputL = 1/(1+np.exp(-self.beta*(np.dot(hiddenL, self.output_l_weights)))) #activation with sigmoid function

		return hiddenL, outputL
		

	def confusion(self, inputs, targets):
		#include bias on the inputs
		inputb = np.ones((inputs.shape[0], 1))*self.bias
		inputb2 = np.concatenate((inputb, inputs), axis=1)

		hiddenL, predicted = self.forward(inputb2)

		average  = 0
		
		if self.outputs ==1:
			predicted = np.array(predicted>=0.5, dtype=int )
		else:
			predicted =np.array(predicted==np.array([np.max(predicted, axis=1)]).T)	

		matrix = np.zeros((self.outputs, self.outputs))	

		
		for targetrow, outputrow in zip(targets,predicted):
			targetindex = np.where(targetrow==1)[0][0]
			outputindex = np.where(outputrow==1)[0][0]
			matrix[targetindex, outputindex]+=1

	
		print(' Confusion matrix \n')
		print(matrix)
			
		for i, k in enumerate(matrix.transpose()):
			value = (k[i]/np.sum(k))*100
			if math.isnan(value):
				value = 0
				
			print("Class Prediction Percentages {} : {}%".format(i+1,value))
			average+=value
				
		print('The average percentage is {}%\n'.format(average/8))
		
		