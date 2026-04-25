from DataPoint import DataPoint, Epoch
from DataLoader import DataLoader
from Transforms import *
import torch.nn as nn
import torchmetrics
import numpy as np
import pickle
import torch
import os

## The only reason I have this hardcoded this way is because I do not currently
## have access to mps or a gpu or any way to access my gpu
DEVICE = "cpu"

class TrainingModel(nn.Module):
	def __init__(self):
		super().__init__()
		self.network = nn.Sequential(
			nn.Conv2d(1, 8, kernel_size=5),
			nn.SiLU(),
			nn.MaxPool2d(kernel_size=2, stride=2),
			
			nn.Conv2d(8, 16, kernel_size=5),
			nn.ReLU(0.1),
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Flatten(),
			
			nn.Linear(400, 120),
			nn.SiLU(),
			nn.Linear(120, 84),
			nn.LeakyReLU(0.1),
			nn.Linear(84, 10) ## 10 digit output
		)
	
	def forward(self, X):
		return self.network(X)


def pointsToTensor(points: list[DataPoint]):
	inputs = []
	targets = []
	
	for point in points:
		inputs.append(point.input)
		targets.append(point.expected)
	
	X = torch.tensor(np.array(inputs), dtype=torch.float32).unsqueeze(1)
	y = torch.tensor(np.array(targets), dtype=torch.long)
	
	return X.to(DEVICE), y.to(DEVICE)


def train(model: TrainingModel, epochs: list[Epoch], testData: list[DataPoint], learnRate: float = 0.0005, decayStep: int = 15, gamma: float = 0.875):
	model.to(DEVICE)
	
	lossFn: nn.Module = nn.CrossEntropyLoss()
	optimizer: torch.optim.Optimizer = torch.optim.Adam(
		model.parameters(),
		lr=learnRate
	)
	scheduler = torch.optim.lr_scheduler.StepLR(
		optimizer,
		step_size = decayStep,
		gamma=gamma
	)
	
	accuracy = torchmetrics.Accuracy(
		task="multiclass",
		num_classes=10
	).to(DEVICE)
	
	for idx in range(len(epochs)):
		model.train()
		X, y = pointsToTensor(epochs[idx].points)
		
		pred: torch.Tensor = model(X)
		loss: torch.Tensor = lossFn(pred, y)
		
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		scheduler.step()
		
		epochLoss = float(loss.item())
		print(f"{idx+1}\t{epochLoss}", end="\t")
		
		model.eval()
		accuracy.reset()
		
		with torch.no_grad():
			Xtest, ytest = pointsToTensor(testData)
			pred: torch.Tensor = model(Xtest)
			accuracy.update(pred, ytest)
		
		acc = float(accuracy.compute().item())
		print(f"{acc}")

def saveModel(fileName: str, model: TrainingModel) -> None:
	with open(fileName, "wb") as f:
		pickle.dump(model.state_dict(), f)

def loadModel(fileName: str) -> TrainingModel:
	model = TrainingModel()
	with open(fileName, 'rb') as f:
		model = pickle.load(f)
	model.eval() ## Switch to evaluation mode
	return model


def genConfusionMtrx(model, epochs: list[Epoch], testData):
	mtrx = np.zeros((10,10), dtype=np.uint64)
	pointLst = []
	
	for ep in epochs:
		pointLst.extend(ep.points)
	
	for tstPt in testData:
		pointLst.append(tstPt)
	
	with torch.no_grad():
		for point in pointLst:
			x = torch.tensor(point.input, dtype=torch.float32).to(DEVICE)
			
			if (x.ndim == 2):
				x = x.unsqueeze(0).unsqueeze(0)
			elif (x.ndim == 3):
				x = x.unsqueeze(0)
			
			out = model(x)
			mtrx[point.expected][torch.argmax(out, dim=1).item()] += 1
	
	lines = []
	for i in range(mtrx.shape[0]):
		lines.append('\t'.join([str(mtrx[i][j]) for j in range(mtrx.shape[1])]))
	return '\n'.join(lines)



def main():
	numAdjs = 4
	adjsPerImg = 5
	useAdjusts = True
	
	trainSplit = 0.8
	batchSize = 20
	learnRate = 0.0078125
	decayStep = 20
	decayRate = 29/32.0
	
	includeHand = True
	includeFont = False
	includeMisc = False
	includeTrainMNIST = True
	includeTestMNIST = True
	
	splitHand = True
	
	numTrainMNIST = 150
	numTestMNIST = 120
	
	
	loader = DataLoader(
		numAdjs,
		adjsPerImg,
		(shift, -4, 4),
		(noise, 3/64, 2.5),
		(brighten, 0.765625, 1.5),
		(rotate, -22.5, 22.5),
		(sharpness, 15/16, 17/16),
		(contrast, 15/16, 17/16)
	)
	loader.loadData(
		trainSplit = trainSplit,
		includeHand = includeHand,
		includeFont = includeFont,
		includeMisc = includeMisc,
		includeTrainMNIST = includeTrainMNIST,
		includeTestMNIST = includeTestMNIST,
		splitHand = splitHand,
		numTrainMNIST = numTrainMNIST,
		numTestMNIST = numTestMNIST
	)
	
	epochs = loader.getEpochs(
		epochSize = batchSize,
		includeAdjusted = useAdjusts
	)
	testData = loader.getTestData(
		includeAdjusted = useAdjusts
	)
	model = TrainingModel()
	
	train(
		model = model, 
		epochs = epochs, 
		testData = testData, 
		learnRate = learnRate, 
		decayStep = decayStep, 
		gamma = decayRate
	)

	modelNum = 1
	modelPath = f"SavedModels/COMB_ADJ/Model_H-{modelNum}.pkl"
	while (os.path.exists(modelPath)):
		modelNum += 1
		modelPath = f"SavedModels/COMB_ADJ/Model_H-{modelNum}.pkl"
	saveModel(modelPath, model)

if (__name__ == "__main__"):
	main()
