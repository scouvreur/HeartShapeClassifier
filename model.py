import numpy as np
from scipy import io as spio
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import svm

def readData():
	'''
	This function reads in the data
	'''
	global HR
	global sysDuration
	global labels
	global features

	data = spio.loadmat('data.mat')

	HR = data['HR']
	sysDuration = data['sysDuration']
	labels = data['labels']

	# Features has the following 12 columns
	# lengthED,lengthES,areaED,areaES,GLS,deltaArea,curvApicalED,
	# curvApicalES,curvSeptalED,curvSeptalES
	features = np.loadtxt('features.csv', delimiter=',')

readData()

all = np.concatenate((features, labels), axis=1)
all = all[~np.isnan(all).any(axis=1)]

X_train = all[:,:12]
Y_train = all[:,12]
Y_train = Y_train.tolist()

X_train, X_validation, Y_train, Y_validation = train_test_split(
    features, labels, test_size=0.4, random_state=747)

# clf = svm.SVC(kernel='linear', C=1)
# clf.fit(X_train, Y_train)
# clf.score(X_validation, Y_validation)