import os

IMAGE_TYPES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

def getExtension(fileName):
    return fileName[fileName.rfind("."):]

def list_images(rootPath):
    imagePaths = []
    for (dirpath, _, filenames) in os.walk(rootPath):
        for filename in filenames:
            if getExtension(filename).endswith(IMAGE_TYPES):
                imagePath = os.path.join(dirpath, filename) 
                imagePaths.append(imagePath)
    
    return imagePaths