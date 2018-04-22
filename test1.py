

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Conv3D, MaxPooling3D
from keras import backend as K
import pickle as pkl
from timeit import default_timer as timer

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from keras_helper import NNWeightHelper
from snes import SNES
import matplotlib.pyplot as plt
import matplotlib.cm as cm




# use just a small sample of the train set to test
SAMPLE_SIZE = 1024
# how many different sets of weights ask() should return for evaluation
POPULATION_SIZE = 10
# how many times we will loop over ask()/tell()
GENERATIONS = 30

def train_classifier(model, X, y):
    X_features = model.predict(X)
    #clf = ExtraTreesClassifier(n_estimators=100, n_jobs=4)
    #clf = DecisionTreeClassifier()
    clf = RandomForestClassifier(n_estimators=100, n_jobs=4)

    clf.fit(X_features, y)
    y_pred = clf.predict(X_features)
    return clf, y_pred


def predict_classifier(model, clf, X):
    X_features = model.predict(X)
    return clf.predict(X_features)

# input image dimensions
img_rows, img_cols = 28, 28
num_classes = 10
	
# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
mnistm = pkl.load(open('mnistm_data.pkl', 'rb'))
mnistm_train = mnistm['train']
mnistm_test = mnistm['test']
mnistm_valid = mnistm['valid']

mnistm_train = mnistm_train.reshape(55000*3,28,28,1)
mnistm_valid = mnistm_valid.reshape(5000*3,28,28,1)
mnistm_test = mnistm_test.reshape(10000*3,28,28,1)

input_shape = (img_rows, img_cols, 1)

b= np.zeros((165000-60000), dtype = int)
y_train = np.concatenate((y_train,b), axis=0)

c = np.zeros((20000), dtype = int)
y_test = np.concatenate((y_test,c), axis=0)

valid_train = np.zeros((15000) , dtype = int)
for i in range(0, 15000):
    valid_train [i] = y_train[i]

mnistm_train = mnistm_train.astype('float32')
mnistm_test = mnistm_test.astype('float32')
mnistm_valid = mnistm_valid.astype('float32')
mnistm_train /= 255
mnistm_test /= 255
mnistm_valid /= 255

print('mnistm_train shape:', mnistm_train.shape)
print(mnistm_train.shape[0], 'train samples')
print(mnistm_test.shape[0], 'test samples')


model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
# model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(1, activation='relu'))
#model.summary()

# this is irrelevant for what we want to achieve
model.compile(loss="mse", optimizer="adam")
#model.fit(mnistm_train,y_train,
#	epochs = 30,batch_size=30,
#	validation_data=(mnistm_valid,valid_train))

print("compilation is over")

nnw = NNWeightHelper(model)
weights = nnw.get_weights()

def main():
	print("Total number of weights to evolve is:", weights.shape)

	all_examples_indices = list(range(mnistm_train.shape[0]))

	clf, _ = train_classifier(model, mnistm_train, y_train)
	print("mnistm_train shape is :", mnistm_train.shape)
	print("y_train shape is :",y_train.shape)
	y_pred = predict_classifier(model, clf, mnistm_test)
	print(y_test.shape, y_pred.shape)
	test_accuracy = accuracy_score(y_test, y_pred)

	print('Non-trained NN Test accuracy:', test_accuracy)
	snes = SNES(weights, 1, POPULATION_SIZE)
	
	for i in range(0, GENERATIONS):
		start = timer()
		asked = snes.ask()

        	# to be provided back to snes
		told = []
        	# use a small number of training samples for speed purposes
		subsample_indices = np.random.choice(all_examples_indices, size=SAMPLE_SIZE, replace=False)
        	# evaluate on another subset
		subsample_indices_valid = np.random.choice(all_examples_indices, size=SAMPLE_SIZE + 1, replace=False)
		#subsample_indices_valid  = mnistm_valid
        	# iterate over the population
		for asked_j in asked:
         		# set nn weights
			nnw.set_weights(asked_j)
			# train the classifer and get back the predictions on the training data
			clf, _ = train_classifier(model, mnistm_train[subsample_indices], y_train[subsample_indices])

			# calculate the predictions on a different set
			y_pred = predict_classifier(model, clf, mnistm_train[subsample_indices_valid])
			score = accuracy_score(y_train[subsample_indices_valid], y_pred)

			# clf, _ = train_classifier(model, x_train, y_train)
			# y_pred = predict_classifier(model, clf, x_test)
			# score = accuracy_score(y_test, y_pred)
			# append to array of values that are to be returned
			told.append(score)

		snes.tell(asked, told)
		end = timer()
		print("It took", end - start, "seconds to complete generation", i + 1)

	nnw.set_weights(snes.center)

	clf, _ = train_classifier(model, mnistm_train, y_train)
	y_pred = predict_classifier(model, clf, mnistm_test)

	print(y_test.shape, y_pred.shape)
	test_accuracy = accuracy_score(y_test, y_pred)

	print('Test accuracy:', test_accuracy)
	cm = confusion_matrix(y_test, y_pred)
	plt.matshow(cm)
	plt.title("confusion matrix")
	plt.colorbar()
	plt.ylabel("True label")
	plt.xlabel("Predicted label")
	plt.show()
	
	


if __name__ == '__main__':
	main()
