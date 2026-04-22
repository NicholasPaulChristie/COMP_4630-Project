from pathlib import Path
from PIL import Image
import numpy as np
import glob
import os

class MyIO:
	def getFiles(self, folderName: str, fileTypes: str | list[str], nFiles: int = None) -> list[str]:
		folder = Path(folderName if folderName else os.getcwd())
		
		try:
			## List all files in the directory
			if (isinstance(fileTypes, str)):
				files = [str(f).replace("\\", "/") for f in list(folder.rglob(f"*{fileTypes}"))]
				for i in range(len(files)):
					while (files[i].find('\\') > -1):
						files[i].replace('\\', '/')
			elif (isinstance(fileTypes, list) and all(isinstance(fType, str) for fType in fileTypes)):
				files = []
				for fileT in fileTypes:
					files.extend([str(f).replace("\\", "/") for f in list(folder.rglob(f"*{fileT}"))])
			
			## Filter and return files based on the given fileType
			if (nFiles is not None):
				if (len(files) > nFiles):
					files = files[:nFiles]
			return files
		except FileNotFoundError:
			print(f"Error: The folder '{folder}' does not exist.")
			return None
		except PermissionError:
			print(f"Error: Permission denied to access '{folder}'.")
			return None
	
	def getFolders(self, folderName: str) -> list:
		folder = folderName if folderName else os.getcwd()
		
		try:
			## List all folders in the directory
			folders = [
				f"{folder}{f}/"
				for f in os.listdir(folder)
				if os.path.isdir(os.path.join(folder, f))
			]
			return folders
		except FileNotFoundError:
			print(f"Error: The folder '{folder}' does not exist.")
			return []
		except PermissionError:
			print(f"Error: Permission denied to access '{folder}'.")
			return []
	
	def readImage(self, fileName: str, convertNP: bool = False):
		img = Image.open(fileName).convert('L') ## 'L' converts to grayscale
		if (convertNP):
			img = np.array(img, dtype=np.uint8)
		return img
	
	#def saveImage(self, fileName: str, )

	def deleteFile(self, fileName: str) -> int:
		try:
			if (os.path.exists(fileName)):
				os.remove(fileName)
				return 1
			return 0
		except PermissionError:
			print(f"Unable to delete '{fileName}' due to permission errors")
			return 0
		except Exception as e:
			print(f"Unable to delete '{fileName}' because {e}")
			return 0
	
	def clearDirectory(self, directory: str):
		filesDeleted = 0
		try:
			files = self.getFiles(directory, "")
			for f in files:
				#self.deleteFile(f)
				filesDeleted += self.deleteFile(f)
		except Exception as e:
			print(f"Unable to delete files in '{directory}' because {e}")
		return filesDeleted
