from typing import Iterable
import numpy as np

class DataPoint:
	__slots__ = ("input", "expected", "name", "sourceName")
	
	def __init__(self, input, expectedIdx: int | None, imgName: str, sourceName: str) -> None:
		self.input = input
		self.expected = expectedIdx
		self.name = imgName
		self.sourceName = sourceName
	
	def __str__(self) -> str:
		return '\n'.join([
			f"Expected: {self.expected}",
			f"Image name: {self.name}",
			f"Source name: {self.sourceName}",
		])
	
	def normalize(self):
		if (isinstance(self.input, np.ndarray)):
			self.input /= 255.0
		else:
			self.input = np.array(self.input, dtype=float) / 255.0

class Epoch:
	__slots__ = ("points")
	
	def __init__(self, epochPoints: Iterable[DataPoint] | None = None):
		self.points = []
		if (epochPoints is not None):
			self.points = [pt for pt in epochPoints if isinstance(pt, DataPoint)]
	
	def __len__(self) -> int:
		return len(self.points)
	
	def add(self, point: DataPoint | Iterable[DataPoint]) -> None:
		if (isinstance(point, DataPoint)):
			self.points.append(point)
		elif (isinstance(point, Iterable)):
			self.points.extend([pt for pt in point if isinstance(pt, DataPoint)])
	
	def shuffle(self) -> None:
		np.random.shuffle(self.points)
	
	def reset(self) -> None:
		self.points.clear()
	
	def normalize(self):
		for point in self.points:
			point.normalize()
