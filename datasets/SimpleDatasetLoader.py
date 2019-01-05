import cv2
import numpy as np
import os

class SimpleDatasetLoader:
    def __init__(self, preprocessors=[]):
        self.preprocessors = preprocessors

        if self.preprocessors is None:
            self.preprocessors = []
        
    def load(self, dataPaths, verbose=-1):

        data = []
        labels = []
        erros = []

        for(i, imagePath) in enumerate(dataPaths):

            image = cv2.imread(imagePath)
            label = imagePath.split(os.path.sep)[-2]
            
            for p in self.preprocessors:
                image = p.preprocess(image)

            data.append(image)
            labels.append(label)
                
            if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
                print("[INFO] processed {}/{}".format(i + 1, len(dataPaths)))

        return (np.array(data), np.array(labels))
