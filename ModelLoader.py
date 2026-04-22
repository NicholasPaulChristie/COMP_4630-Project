import torch.nn as nn


MODEL_IDXS = {
	"A": 0,
	"B": 1,
	"C": 2,
	"D": 3,
	"E": 4,
	"F": 5,
	"G": 6,
	"H": 7,
}

def loadModelFrame(modelChar: str = 'H'):
	return [
		nn.Sequential( ## Model A
			nn.Conv2d(1, 32, kernel_size=5, padding=2),
			nn.ReLU(),
			nn.Conv2d(32, 32, kernel_size=3, padding=1),
			nn.ReLU(),
			
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Conv2d(32, 32, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Flatten(),
			
			nn.LazyLinear(128),
			nn.Linear(128, 96),
			nn.Linear(96, 80),
			nn.LazyLinear(10) ## 10 digit output
		),
		nn.Sequential( ## Model B
			nn.Conv2d(1, 32, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.Conv2d(32, 32, kernel_size=3, padding=1),
			nn.ReLU(),
			
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Conv2d(32, 32, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Flatten(),
			
			nn.LazyLinear(128),
			nn.Linear(128, 96),
			nn.Linear(96, 80),
			nn.Linear(80, 80),
			nn.Linear(80, 72),
			nn.Linear(72, 72),
			nn.Linear(72, 68),
			nn.LazyLinear(10) ## 10 digit output
		),
		nn.Sequential( ## Model C
			nn.Conv2d(1, 32, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.Conv2d(32, 32, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Conv2d(32, 32, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Flatten(),
			
			nn.LazyLinear(128),
			nn.ReLU(),
			nn.Dropout(0.5),
			nn.Linear(128, 64),
			nn.ReLU(),
			nn.Linear(64, 10) ## 10 digit output
		),
		nn.Sequential( ## Model D
			nn.Conv2d(1, 32, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.Conv2d(32, 32, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Conv2d(32, 32, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Flatten(),
			
			nn.Linear(2048, 256),
			nn.SiLU(),
			nn.Linear(256, 128),
			nn.Softplus(),
			nn.Linear(128, 10) ## 10 digit output
		),
		nn.Sequential( ## Model E
			nn.Conv2d(1, 32, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.Conv2d(32, 32, kernel_size=3, padding=1),
			nn.ReLU(),
			
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Conv2d(32, 32, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Flatten(),
			
			nn.Linear(2048, 128),
			nn.LeakyReLU(0.1),
			nn.Linear(128, 64),
			nn.ELU(),
			nn.Linear(64, 10) ## 10 digit output
		),
		nn.Sequential( ## Model F
			nn.Conv2d(1, 6, kernel_size=5), ## 32 -> 28
			nn.Tanh(),
			nn.AvgPool2d(2), ## 28 -> 14
			
			nn.Conv2d(6, 16, kernel_size=5), ## 14 -> 10
			nn.Tanh(),
			nn.AvgPool2d(2), ## 10 -> 5
			
			nn.Flatten(),
			
			nn.Linear(400, 120),
			nn.Tanh(),
			nn.Linear(120, 84),
			nn.Tanh(),
			nn.LazyLinear(10) ## 10 digit output
		),
		nn.Sequential( ## Model G
			nn.Conv2d(1, 6, kernel_size=5),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2),
			
			nn.Conv2d(6, 16, kernel_size=5),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2),
			nn.Flatten(),
			
			nn.Linear(400, 120),
			nn.ReLU(),
			nn.Linear(120, 84),
			nn.ReLU(),
			nn.LazyLinear(10) ## 10 digit output
		),
		nn.Sequential( ## Model H
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
	][MODEL_IDXS[modelChar]]
