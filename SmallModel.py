from ModelLoader import loadModelFrame
from DataLoader import DataLoader
from DataPoint import DataPoint
from MyIO import MyIO
from PIL import Image
import torch.nn as nn
import torchmetrics
import numpy as np
import pickle
import torch

import glob
import os

IO = MyIO()
DEVICE = "cpu"

class SmolBoi(nn.Module):
	def __init__(self, modelChar: str = "A"):
		super().__init__()
		self.network = loadModelFrame(modelChar)
	
	def forward(self, X):
		return self.network(X)


def sortData(dataPoints: list[DataPoint]):
	dataLsts = {}
	
	## Group by source
	for point in dataPoints:
		try:
			dataLsts[point.sourceName].append(point)
		except KeyError:
			dataLsts |= {point.sourceName: [point]}
	
	sources = list(dataLsts)
	for source in sources:
		points = [[] for _ in range(10)]
		for pnt in dataLsts[source]:
			points[pnt.expected].append(pnt)
		
		for i in range(10):
			points[i] = sortDigitIdx(points[i])
		
		## Flatten list
		dataLsts[source] = [item for subLst in points for item in subLst]
	return dataLsts

def sortDigitIdx(arr):
	## Sorts data by digit index
	if (len(arr) <= 1):
		return arr
	
	pivot = int(arr[len(arr) // 2].name.split('-')[-1])
	
	left, mid, right = [], [], []
	for item in arr:
		val = int(item.name.split('-')[-1])
		if (val < pivot):
			left.append(item)
		elif (val > pivot):
			right.append(item)
		else:
			mid.append(item)
	return sortDigitIdx(left) + mid + sortDigitIdx(right)


def evalDataGroup(model: SmolBoi, points: list[DataPoint]):
	model.to(DEVICE)
	model.eval()
	
	accuracy = torchmetrics.Accuracy(task="multiclass", num_classes = 10).to(DEVICE)
	precision = torchmetrics.Precision(task="multiclass", num_classes = 10, average = "macro").to(DEVICE)
	specificity = torchmetrics.Specificity(task="multiclass", num_classes = 10, average = "macro").to(DEVICE)
	recall = torchmetrics.Recall(task="multiclass", num_classes = 10, average = "macro").to(DEVICE)
	criterion = nn.CrossEntropyLoss()
	totalLoss = 0.0
	
	with torch.no_grad():
		for point in points:
			if (point.expected is None):
				continue
			elif (point.expected > 10 or point.expected < 0):
				continue
			
			## Convert input to tensor
			X = torch.from_numpy(point.input).float()
			
			## Ensure correct shape: (1, 1, H, W)
			if (X.ndim == 2):
				X = X.unsqueeze(0).unsqueeze(0)
			elif (X.ndim == 3):
				X = X.unsqueeze(0)
			
			y = torch.tensor([point.expected], dtype=torch.long)
			
			## Forward pass
			logits = model(X)
			
			## Loss
			loss = criterion(logits, y)
			totalLoss += loss.item()
			
			## Predictions
			preds = torch.argmax(logits, dim=1)
			
			## Update metrics
			accuracy.update(preds, y)
			precision.update(preds, y)
			specificity.update(preds, y)
			recall.update(preds, y)
		
		## Compute final metrics
		acc = accuracy.compute().item()
		prec = precision.compute().item()
		spec = specificity.compute().item()
		rec = recall.compute().item()
		
		# print(f"Accuracy: {acc:.5f}")
		# print(f"Precision: {prec:.5f}")
		# print(f"Specificity: {spec:.5f}")
		# print(f"Recall: {rec:.5f}")
		# print(f"Total loss: {totalLoss:.5f}")
		# print(f"Average loss: {totalLoss/len(points):.5f}")
		
		return acc, prec, spec, rec, totalLoss, totalLoss/len(points)
	#printConfusionMtrx(genConfusionMtrx(model, points))
	print()



def dataLstToTensor(data: list[DataPoint]):
	inputs = []
	targets = []

	for point in data:
		inputs.append(point.input)
		targets.append(point.expected)
	
	X = torch.tensor(np.array(inputs), dtype=torch.float32).unsqueeze(1)
	y = torch.tensor(np.array(targets), dtype=torch.long)

	return X.to(DEVICE), y.to(DEVICE)

def pointToTensor(point: DataPoint):
	return torch.from_numpy(point.input).float()


def loadModel(fileName: str, modelChar: str) -> SmolBoi:
	model = SmolBoi(modelChar)
	with open(fileName, "rb") as f:
		model.load_state_dict(pickle.load(f))
	model.eval() ## Set to evaluation mode by default
	
	return model

def loadModels(modelChar: str = "H", trainType: str = "COMBINED") -> list[SmolBoi]:
	if (trainType not in ["COMBINED", "HAND_ONLY", "MNIST_ONLY", "COMB_ADJ"]):
		raise ValueError(f"Invalid trainType of {trainType}. Please pass 'COMBINED', 'HAND_ONLY', or 'MNIST_ONLY'")
	
	modelFolder = f"SavedModels/MODEL_{modelChar}/{trainType}/"
	baseFiles = IO.getFiles(modelFolder, ".pkl")
	
	modelDict = {
		int(file.split('-')[-1].split('.')[0]): file
		for file in baseFiles
	}
	
	return [
		loadModel(modelDict[idx], modelChar)
		for idx in sorted(list(modelDict))
	]


def genConfusionMtrx(model, data: list[DataPoint]):
	mtrx = np.zeros((10,10), dtype=np.uint64)
	
	with torch.no_grad():
		for point in data:
			x = torch.tensor(point.input, dtype=torch.float32).to(DEVICE)
			
			if (x.ndim == 2):
				x = x.unsqueeze(0).unsqueeze(0)
			elif (x.ndim == 3):
				x = x.unsqueeze(0)
			
			out = model(x)
			mtrx[point.expected][torch.argmax(out, dim=1).item()] += 1
			# if (point.expected != torch.argmax(out, dim=1).item()):
			# 	print(f"{point.source}: {point.name} -> {torch.argmax(out, dim=1).item()}")
	return mtrx

def printConfusionMtrx(mtrx: np.ndarray) -> str:
	lines = [("\t" + '\t'.join([str(i) for i in range(10)]))]
	for i in range(mtrx.shape[0]):
		row = [(str(i) + ":")]
		for j in range(mtrx.shape[1]):
			row.append(str(mtrx[i][j]))
		lines.append('\t'.join(row))
	return '\n'.join(lines)

def saveConfusionMatrix(model, data):
	N = 25
	
	mtrx = genConfusionMtrx(model, data)
	with open("Matrix.txt", 'w') as f:
		f.write(printConfusionMtrx(mtrx))
	minV = np.min(mtrx)
	maxV = np.max(mtrx)
	
	rounded = (mtrx-minV)*255/(maxV-minV)
	rounded = rounded.astype(np.uint8)
	bigRound = np.zeros((10*N,10*N), dtype = rounded.dtype)
	
	for y in range(rounded.shape[0]):
		for x in range(rounded.shape[1]):
			for a in range(N):
				for b in range(N):
					bigRound[N*y+a][N*x+b] = rounded[y][x]
	
	img = Image.fromarray(bigRound.astype(np.uint8))
	img.save("Matrix.png")


def classifyPoints(model: SmolBoi, data: list[DataPoint]):
	sortedData = sortData(data)
	sources = sorted(list(sortedData))
	
	totalDeleted = 0
	for digit in range(10):
		totalDeleted += IO.clearDirectory(f"Classified/{digit}/")

	with torch.no_grad():
		for source in sources:
			if (source in ["Misc",]):
				continue
			
			for point in sortedData[source]:
				x = torch.tensor(point.input, dtype=torch.float32).to(DEVICE)
				
				if (x.ndim == 2):
					x = x.unsqueeze(0).unsqueeze(0)
				elif (x.ndim == 3):
					x = x.unsqueeze(0)
				
				out = model(x)
				pred = torch.argmax(out, dim=1).item()
				if (point.expected != pred):
					img = Image.fromarray((point.input * 255).astype(np.uint8))
					img.save(f"Classified/{pred}/{point.expected}-{point.name} ({point.sourceName}).png")
					# print(f"{point.sourceName}: {point.name} -> {pred}")


def showAllModels(data: list[DataPoint]):
	accuracyHist = []
	precisionHist = []
	specificityHist = []
	recallHist = []
	totalLossHist = []
	aveLossHist = []
	
	modelFrame = 'H'
	typeTrain = ["COMBINED", "HAND_ONLY", "MNIST_ONLY", "COMB_ADJ"]
	typeTrainIdx = 0-1
	models = loadModels(modelFrame, typeTrain[typeTrainIdx])
	
	for mdl in models:
		acc, prec, spec, rec, tLoss, aLoss = evalDataGroup(mdl, data)
		accuracyHist.append(acc)
		precisionHist.append(prec)
		specificityHist.append(spec)
		recallHist.append(rec)
		totalLossHist.append(tLoss)
		aveLossHist.append(aLoss)
	print("Here")
	for i in range(len(accuracyHist)):
		line = [
			f"Model {i+1}",
			str(accuracyHist[i]),
			str(precisionHist[i]),
			str(specificityHist[i]),
			str(recallHist[i]),
			str(totalLossHist[i]),
			str(aveLossHist[i])
		]
		print('\t'.join(line))
	input("Done")


def main():
	loader = DataLoader()
	
	includeHand = True
	includeFont = False
	includeMisc = False
	includeTrainMNIST = False
	includeTestMNIST = False
	includeAI = False
	
	numTrainMNIST = -1
	numTestMNIST = -1
	
	## Load model
	#model = loadModel("SavedModels/Model_H/COMB_ADJ/MODEL_H-10.pkl", 'H')
	
	## Load data
	loader.loadData(
		includeHand = includeHand,
		includeFont = includeFont,
		includeMisc = includeMisc,
		includeTrainMNIST = includeTrainMNIST,
		includeTestMNIST = includeTestMNIST,
		includeAI = includeAI,
		splitHand = False,
		numTrainMNIST = numTrainMNIST,
		numTestMNIST = numTestMNIST
	)
	data = loader.getLoadedData(flatten=True, includeAdjusted=False)
	showAllModels(data)
	
	#evalDataGroup(model, data)
	#classifyPoints(model, data)
	
	# groupData = sortData(data)
	
	# for name, lst in groupData.items():
	# 	print(name)
	# 	acc, prec, spec, rec, tLoss, aLoss = evalDataGroup(model, lst)
	# 	print(f"Accuracy: {acc:.5f}")
	# 	print(f"Precision: {prec:.5f}")
	# 	print(f"Specificity: {spec:.5f}")
	# 	print(f"Recall: {rec:.5f}")
	# 	print(f"Total loss: {tLoss:.5f}")
	# 	print(f"Average loss: {aLoss:.5f}", end="\n\n")
	
	# if (len(list(groupData)) > 1):
	# 	acc, prec, spec, rec, tLoss, aLoss = evalDataGroup(model, data)
		
	# 	print("Overall")
	# 	print(f"Accuracy: {acc:.5f}")
	# 	print(f"Precision: {prec:.5f}")
	# 	print(f"Specificity: {spec:.5f}")
	# 	print(f"Recall: {rec:.5f}")
	# 	print(f"Total loss: {tLoss:.5f}")
	# 	print(f"Average loss: {aLoss:.5f}")
	# input()

if (__name__ == "__main__"):
	main()
