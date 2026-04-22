from DataPoint import DataPoint, Epoch
from Transforms import *
from pathlib import Path
from PIL import Image
import numpy as np
import os
from MyIO import MyIO

IO = MyIO()

NUM_DIGITS = 10
IMAGE_SIDE = 32
MNIST_TRAIN_COUNTS = {
	0: 5923, 
	1: 6742, 
	2: 5958, 
	3: 6131, 
	4: 5841, 
	5: 5421, 
	6: 5918, 
	7: 6266, 
	8: 5851, 
	9: 5949
}
MNIST_TEST_COUNTS = {
	0: 980, 
	1: 1135, 
	2: 1032, 
	3: 1010, 
	4: 982, 
	5: 892, 
	6: 958, 
	7: 1028, 
	8: 974, 
	9: 1009
}

class DataLoader:
	"""
	__slots__ = (
		"_trainData",
		"_testData",
		"_trainMNIST",
		"_testMNIST",
		"_font",
		"_misc",
		"_chatData",
		"_numAdjusts",
		"_maxAdjustsPerImg",
		"_adjustments",
		"_adjTrain",
		"_adjTest",
	)
	"""
	_trainData = None
	_testData = None
	
	_trainMNIST = None
	_testMNIST = None
	
	_numAdjusts = None
	_maxAdjustsPerImg = None
	_adjustments = None
	_adjTrain = None
	_adjTest = None
	
	_font = None
	_misc = None
	_chatData = None
	
	
	def __init__(self, numAdjs: int | None = None, maxAdjsPerImg: int | None = 4, *adjustments) -> None:
		if (numAdjs is not None):
			self._numAdjusts = numAdjs
			self._maxAdjustsPerImg = max(1, maxAdjsPerImg)
			self._adjustments = adjustments
			self._adjTrain = []
			self._adjTest = []
	
	def loadData(
			self,
			includeHand: bool = True,
			includeFont: bool = False, ## Not used in model training
			includeMisc: bool = False, ## Not used in model training
			includeAI: bool = False, ## Not used in model training
			includeTrainMNIST: bool = True,
			includeTestMNIST: bool = True,
			trainSplit: float = 0.75,
			splitHand: bool = True,
			numTrainMNIST: int = 250,
			numTestMNIST: int = 200):
		## Load Personal Data
		if (includeHand):
			self._trainData = [[] for _ in range(NUM_DIGITS)]
			self._testData = [[] for _ in range(NUM_DIGITS)]
			
			files = IO.getFiles(f"Data/Hand/", ".png")
			for file in files:
				#print(f"Loading '{file}'", end="\t")
				tmp = [[] for _ in range(NUM_DIGITS)]
				image = IO.readImage(file, True)
				name = (file.split('/')[2:])[0].split('.')[0]
				
				## Chop input image into points
				rows = image.shape[0] // IMAGE_SIDE
				cols = image.shape[1] // IMAGE_SIDE
				for y in range(rows):
					yOff = y * IMAGE_SIDE
					for x in range(cols):
						xOff = x*IMAGE_SIDE
						img = image[yOff:yOff+IMAGE_SIDE, xOff:xOff+IMAGE_SIDE]
						
						if (np.sum(img) > 0):
							img = Image.fromarray(img)
							tmp[y].append(
								DataPoint(
									input = img, 
									expectedIdx = y, 
									imgName = f"{x}", 
									sourceName = name
								)
							)
				
				for i in range(NUM_DIGITS):
					if (splitHand):
						splitIdx = int(len(tmp[i]) * trainSplit)
						idxs = np.arange(len(tmp[i]))
						np.random.shuffle(idxs)
						
						idxs = idxs[:splitIdx]
						mask = np.zeros(len(tmp[i]), dtype=bool)
						mask[idxs] = True
						
						for mskIdx in range(len(mask)):
							if (mask[mskIdx]):
								self._trainData[i].append(tmp[i][mskIdx])
							else:
								self._testData[i].append(tmp[i][mskIdx])
					else:
						self._trainData[i].extend(tmp[i])
		
		## Load Font
		if (includeFont):
			print("Loading Font")
			self._font = [[] for _ in range(NUM_DIGITS)]
			image = IO.readImage("Data/Font.png", True)
			
			for y in range(NUM_DIGITS):
				yOff = y * IMAGE_SIDE
				for x in range(image.shape[1]//IMAGE_SIDE):
					xOff = x * IMAGE_SIDE
					self._font[y].append(
						DataPoint(
							input = image[yOff:yOff+IMAGE_SIDE, xOff:xOff+IMAGE_SIDE],
							expectedIdx = y,
							imgName = f"{x}",
							sourceName = "Font"
						)
					)
		
		## Load Misc "points"
		if (includeMisc):
			print("Loading misc points")
			folderName = f"Data/Misc/"
			self.misc = []
			
			files = IO.getFiles(folderName, ".png")
			for file in files:
				self.misc.append(
					DataPoint(
						input = IO.readImage(file), 
						expectedIdx = -1, 
						imgName = file.split(folderName)[-1],
						sourceName = "Misc"
					)
				)
		
		if (includeAI):
			self._chatData = [[] for _ in range(NUM_DIGITS)]
			
			files = IO.getFiles("Data/AI/", ".png")
			for file in files:
				#print(f"Loading '{file}'")
				image = IO.readImage(file, True)
				name = (file.split('/')[2:])[0].split('.')[0]
				
				rows = image.shape[0] // IMAGE_SIDE
				cols = image.shape[1] // IMAGE_SIDE
				for y in range(rows):
					yOff = y*IMAGE_SIDE
					for x in range(cols):
						xOff = x*IMAGE_SIDE
						self._chatData[y].append(
							DataPoint(
								input = image[yOff:yOff+IMAGE_SIDE, xOff:xOff+IMAGE_SIDE], 
								expectedIdx = y, 
								imgName = f"{x}", 
								sourceName = name
							)
						)
		
		## Load MNIST training data
		if (includeTrainMNIST):
			#print("Loading MNIST data (training)")
			self._trainMNIST = [[] for _ in range(NUM_DIGITS)]
			image = IO.readImage(f"MNIST/Train.png", True)
			
			for y in range(NUM_DIGITS):
				yOff = y * IMAGE_SIDE
				
				points = np.arange(MNIST_TRAIN_COUNTS[y])
				np.random.shuffle(points)
				
				if (numTrainMNIST > -1):
					points = np.random.choice(points, size=min(MNIST_TRAIN_COUNTS[y], numTrainMNIST), replace=False)
				
				for point in points:
					xOff = point * IMAGE_SIDE
					img = image[yOff:yOff+IMAGE_SIDE, xOff:xOff+IMAGE_SIDE]
					
					if (np.sum(img) > 0):
						img = Image.fromarray(img)
						self._trainMNIST[y].append(
							DataPoint(
								input = img, 
								expectedIdx = y, 
								imgName = f"{point}", 
								sourceName = "MNIST_Train"
							)
						)
		
		## Load MNIST testing data
		if (includeTestMNIST):
			#print("Loading MNIST data (testing)")
			self._testMNIST = [[] for _ in range(NUM_DIGITS)]
			image = IO.readImage(f"MNIST/Test.png", True)
			
			for y in range(NUM_DIGITS):
				yOff = y * IMAGE_SIDE
				
				points = np.arange(MNIST_TEST_COUNTS[y])
				np.random.shuffle(points)
				
				if (numTestMNIST > -1):
					points = np.random.choice(points, size=min(MNIST_TEST_COUNTS[y], numTestMNIST), replace=False)
				
				for point in points:
					xOff = point * IMAGE_SIDE
					img = image[yOff:yOff+IMAGE_SIDE, xOff:xOff+IMAGE_SIDE]
					
					if (np.sum(img) > 0):
						img = Image.fromarray(img)
						self._testMNIST[y].append(
							DataPoint(
								input = img, 
								expectedIdx = y,
								imgName = f"{point}",
								sourceName = "MNIST_Test"
							)
						)
		
		
		## Adjust loaded data
		if (self._numAdjusts is not None and (self._trainData is not None or self._trainMNIST is not None)):
			for digit in range(NUM_DIGITS):
				train = []
				test = []
				
				## Adjust hand training data
				if (self._trainData is not None):
					numPoints = len(self._trainData[digit])
					for ptIdx in range(numPoints):
						train.extend(
							adjust(
								point = self._trainData[digit][ptIdx], 
								numAdjs = self._numAdjusts,
								adjsPerImg = self._maxAdjustsPerImg,
								adjustments = self._adjustments
							)
						)
				
				## Adjust MNIST training data
				if (self._trainMNIST is not None):
					numPoints = len(self._trainMNIST[digit])
					for ptIdx in range(numPoints):
						train.extend(
							adjust(
								point = self._trainMNIST[digit][ptIdx], 
								numAdjs = self._numAdjusts,
								adjsPerImg = self._maxAdjustsPerImg,
								adjustments = self._adjustments
							)
						)
				self._adjTrain.append(train)
				
				
				## Adjust hand testing data
				if (self._testData is not None):
					numPoints = len(self._testData[digit])
					for ptIdx in range(numPoints):
						test.extend(
							adjust(
								point = self._testData[digit][ptIdx], 
								numAdjs = self._numAdjusts,
								adjsPerImg = self._maxAdjustsPerImg,
								adjustments = self._adjustments
							)
						)
				
				## Adjust MNIST testing data
				if (self._testMNIST is not None):
					numPoints = len(self._testMNIST[digit])
					for ptIdx in range(numPoints):
						test.extend(
							adjust(
								point = self._testMNIST[digit][ptIdx], 
								numAdjs = self._numAdjusts,
								adjsPerImg = self._maxAdjustsPerImg,
								adjustments = self._adjustments
							)
						)
				self._adjTest.append(test)
	
	
	def getLoadedData(self, flatten: bool = False, includeAdjusted: bool = False):
		if (self._trainData is None and self._trainMNIST is None and self._testData is None and self._testMNIST is None and self._font is None and self._misc is None and self._chatData is None):
			raise ValueError("Data has not been loaded yet!")
		data = [[] for _ in range(NUM_DIGITS)]
		
		for digit in range(NUM_DIGITS):
			if (self._trainData is not None):
				data[digit].extend(self._trainData[digit])
			
			if (self._testData is not None):
				data[digit].extend(self._testData[digit])
			
			if (self._trainMNIST is not None):
				data[digit].extend(self._trainMNIST[digit])
			
			if (self._testMNIST is not None):
				data[digit].extend(self._testMNIST[digit])
			
			if (self._font is not None):
				data[digit].extend(self._font[digit])
			
			if (self._chatData is not None):
				data[digit].extend(self._chatData[digit])
			
			if (includeAdjusted and self._adjTrain is not None):
				data[digit].extend(self._adjTrain[digit])
			
			if (includeAdjusted and self._adjTest is not None):
				data[digit].extend(self._adjTest[digit])
		
		if (flatten):
			data = self._flatten(data)
			for i in range(len(data)):
				if (type(data[i].input) != np.ndarray):
					data[i].normalize()
		else:
			for digit in range(NUM_DIGITS):
				for i in range(len(data[digit])):
					if (type(data[i][digit].input) != np.ndarray):
						data[i][digit].normalize()
		return data
	
	"""
	def getTrainData(self, flatten: bool = False, includeAdjusted: bool = True):
		'''
		if (not self._dataLoaded):
			raise ValueError("Data has not been loaded yet!")
		'''
		data = self._trainData
		
		if (includeAdjusted and self._adjTrain is not None):
			for i in range(NUM_DIGITS):
				data[i].extend(self._adjTrain)
		
		if (flatten):
			data = self._flatten(data)
		return data
	"""
	
	def getTestData(self, includeAdjusted: bool = False):
		testData = self._flatten(self._testData)
		if (self._testMNIST is not None):
			testData.extend(self._flatten(self._testMNIST))
		
		if (includeAdjusted and self._adjTest is not None):
			for adjDigit in self._adjTest:
				testData.extend(adjDigit)
		
		np.random.shuffle(testData)
		for point in testData:
			point.normalize()
		
		return testData
	
	
	def getFont(self, flatten: bool = True):
		if (self._font is None):
			return []
		
		data = self._font
		if (flatten):
			data = self._flatten(self._font)
		return data
	
	def getMisc(self):
		if (self._misc is None):
			return []
		else:
			return self._misc
	
	def getChatPts(self):
		if (self._chatData is None):
			return []
		else:
			return self._chatData
	
	def getEpochs(self, epochSize: int = 20, includeAdjusted: bool = True, shuffleData: bool = True):
		epochs = []
		last = None

		## Accumulate training data
		data = []
		for i in range(NUM_DIGITS):
			digit = []
			if (self._trainData is not None):
				digit.extend(self._trainData[i])
			
			if (self._trainMNIST is not None):
				digit.extend(self._trainMNIST[i])
			
			if (includeAdjusted and self._adjTrain is not None):
				digit.extend(self._adjTrain[i])
			
			data.append(digit)
		data = self._flatten(data)
		
		## Shuffle data
		if (shuffleData):
			np.random.shuffle(data)
		
		## prune extra data points for later
		remain = len(data) % epochSize
		if (remain > 0):
			last = Epoch(data[len(data)-remain:])
			last.normalize()
			data = data[:len(data)-remain]
		
		## Create epochs
		for i in range(0, len(data), epochSize):
			epochs.append(Epoch(data[i:i+epochSize]))
			epochs[-1].normalize()
		np.random.shuffle(epochs)
		
		if (last is not None):
			print(len(last), "points in last epoch")
			##epochs[len(epochs)-1].add(last)
			epochs.append(last)
		
		return epochs
	
	def _flatten(self, arr):
		return [item for subLst in arr for item in subLst]
