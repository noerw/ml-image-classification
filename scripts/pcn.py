
# Code from Chapter 3 of Machine Learning: An Algorithmic Perspective (2nd Edition)
# by Stephen Marsland (http://stephenmonika.net)

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Stephen Marsland, 2008, 2014

import numpy as np

class Pcn:
	""" A basic Perceptron"""

	def __init__(self,inputs,targets):
		""" Constructor """
		# Set up network size
		if np.ndim(inputs)>1:
			self.nIn = np.shape(inputs)[1] # extract number of inputs (columns)
		else:
			self.nIn = 1

		if np.ndim(targets)>1:
			self.nOut = np.shape(targets)[1]
		else:
			self.nOut = 1

		self.nData = np.shape(inputs)[0] # amount of samples

                # Initialise network: randomized weights table (max +- 0.1)
                # +1 dimension for the bias node w_{0,j} !
		self.weights = np.random.rand(self.nIn+1,self.nOut)*0.1-0.05

	def pcntrain(self,inputs,targets,eta,nIterations):
		""" Train the thing """
                # eta: factor of weight adaption

		# Add the inputs that match the bias node
                # add constant input of -1 to accomodate for all 0 inputs
		inputs = np.concatenate((inputs,-np.ones((self.nData,1))),axis=1)

		# Training
		change = list(range(self.nData))

		for n in range(nIterations):
			self.activations = self.pcnfwd(inputs);
			self.weights -= eta*np.dot(np.transpose(inputs),self.activations-targets)

			# Randomise order of inputs
			np.random.shuffle(change)
			inputs = inputs[change,:]
			targets = targets[change,:]

		return self.weights

	def pcnfwd(self,inputs):
		""" Run the network forward """
		# Compute activations
		activations =  np.dot(inputs,self.weights)

		# Threshold the activations
		return np.where(activations>0,1,0)


	def confmat(self,inputs,targets):
		"""Confusion matrix"""

		# Add the inputs that match the bias node
		inputs = np.concatenate((inputs,-np.ones((self.nData,1))),axis=1)

		outputs = np.dot(inputs,self.weights)

		nClasses = np.shape(targets)[1]

		if nClasses==1:
			nClasses = 2
			outputs = np.where(outputs>0,1,0)
		else:
			# 1-of-N encoding
			outputs = np.argmax(outputs,1)
			targets = np.argmax(targets,1)

		cm = np.zeros((nClasses,nClasses))
		for i in range(nClasses):
			for j in range(nClasses):
				cm[i,j] = np.sum(np.where(outputs==i,1,0)*np.where(targets==j,1,0))

		print(cm)
		print(np.trace(cm)/np.sum(cm))

def testPcn():
	def splitInputsTargets(data, numInputs):
		# training data, first two values: inputs, last value: expected output
		inputs = data[:, 0:numInputs]
		targets = data[:, numInputs:]
		return inputs, targets

	def evalModel(name, data, numInputs, numIterations):
		print('\n {}\twith {} inputs and {} iterations'.format(name, numInputs, numIterations))
		inp, out = splitInputsTargets(data, numInputs)
		p = Pcn(inp, out)
		p.pcntrain(inp, out, 0.25, numIterations)
		p.confmat(inp, out)

	trainAnd = np.array([[0,0,0],[0,1,0],[1,0,0],[1,1,1]])
	trainOr = np.array([[0,0,0],[0,1,1],[1,0,1],[1,1,1]])
	trainXor = np.array([[0,0,0],[0,1,1],[1,0,1],[1,1,0]])

	# XOR with additional dummy input to make the problem linearly separatable. magic!!
	# (called projection, automatable via "kernel classifiers"?)
	trainXor3D = np.array([[1,0,0,0],[0,0,1,1],[0,1,0,1],[0,1,1,0]])

	evalModel('AND', trainAnd, 2, 7)
	evalModel('OR', trainOr, 2, 5)
	evalModel('XOR', trainXor, 2, 7)
	evalModel('XOR3D', trainXor3D, 3, 20)

