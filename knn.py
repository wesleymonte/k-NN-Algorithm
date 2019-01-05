from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from preprocessing.SimplePreprocessor import SimplePreprocessor
from datasets.SimpleDatasetLoader import SimpleDatasetLoader
from myutil import paths
import argparse
import numpy as np

def reshapeImages(Images):
    return np.array([i.reshape(1, 3072) if i is not None else '' for i in Images])

def numOfNones(List):
    num = 0
    for e in List:
        if e is None: num+=1
    return num

def removeNones(List=[]):
    removeIndexs = []
    for i in range(len(List)):
        if List[i] is None:
            removeIndexs.append(i)
    return np.delete(List, removeIndexs)


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="Path to input dataset")
ap.add_argument("-k", "--neighbors", type=int, default=1,
                help="# of nearest neighbors for classification")
ap.add_argument("-j", "--jobs", type=int, default=1,
                help="# of jobs for k-NN distance (-1 uses all available cores)")
args = vars(ap.parse_args())

print("[INFO] loading images...")

imagePaths = paths.list_images(args["dataset"])
sp = SimplePreprocessor(32, 32)
sdl = SimpleDatasetLoader(preprocessors=[sp])
(data, labels) = sdl.load(imagePaths, verbose=500)
data = data.reshape((data.shape[0], 3072))

print("[INFO] features matrix: {:.1f}MB".format(data.nbytes / (1024 * 1000.0)))

le = LabelEncoder()
labels = le.fit_transform(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

print("[INFO] evaluation k-NN classifier...")
print(data.shape)
print(labels.shape)
model = KNeighborsClassifier(n_neighbors=args["neighbors"], n_jobs=args["jobs"])
model.fit(trainX, trainY)
print(classification_report(testY, model.predict(testX),
        target_names=le.classes_))