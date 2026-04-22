from PIL import Image, ImageEnhance
from DataPoint import DataPoint
import numpy as np

def adjust(point: DataPoint, numAdjs: int, adjsPerImg: int, adjustments: list):
	if (adjsPerImg < 0 or adjsPerImg > 6):
		raise ValueError(f"Invalid adjsPerImg value of {adjsPerImg}")
	
	newPoints = []
	for i in range(numAdjs):
		img = point.input.copy()
		adjusts = np.random.choice(
			np.arange(len(adjustments)),
			size = np.random.randint(min(6, adjsPerImg)),
			replace = False
		)
		
		for idx in adjusts:
			func, minVal, maxVal = adjustments[idx]
			
			if (func is shift):
				x = np.random.randint(minVal, maxVal+1)
				y = np.random.randint(minVal, maxVal+1)
				img = func(img, x, y)
			else:
				factor = np.random.random() * (maxVal - minVal) + minVal
				img = func(img, factor)
		
		newPoints.append(
			DataPoint(
				input = img,
				expectedIdx = point.expected,
				imgName = f"{point.name} (adj {i})",
				sourceName = point.sourceName
			)
		)
	return newPoints


def rotate(image: Image.Image, degrees: float):
	return image.rotate(degrees, expand=False)

def shift(image: Image.Image, shiftX: int = 0, shiftY: int = 0):
	w, h = image.size
	shft = Image.new(image.mode, (w, h), 0) ## Create new blank image
	shft.paste(image, (shiftX, shiftY))
	return shft

def noise(image: Image.Image, noiseLvl: float = 1.0) -> np.ndarray:
	arr = np.array(image).astype(np.float64)
	noise = np.random.normal(0, noiseLvl, arr.shape)
	return Image.fromarray(np.clip(arr+noise, 0, 255).astype(np.uint8))

def brighten(image: Image.Image, factor: float = 1.0) -> Image.Image:
	enhancer = ImageEnhance.Brightness(image)
	return enhancer.enhance(factor)

def sharpness(image: Image.Image, factor: float = 1.0) -> Image.Image:
	enhancer = ImageEnhance.Sharpness(image)
	return enhancer.enhance(factor)

def contrast(image: Image.Image, factor: float = 1.0) -> Image.Image:
	enhancer = ImageEnhance.Contrast(image)
	return enhancer.enhance(factor)
