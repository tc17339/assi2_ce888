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
from tensorflow.examples.tutorials.mnist import input_data

# use just a small sample of the train set to test
SAMPLE_SIZE = 1024
# how many different sets of weights ask() should return for evaluation
POPULATION_SIZE = 10
# how many times we will loop over ask()/tell()
GENERATIONS = 30

def train_classifier(model, X, y):
    X_features = model.predict(X)
    clf = ExtraTreesClassifier(n_estimators=100, n_jobs=4)
    #clf = DecisionTreeClassifier()
    clf.fit(X_features, y)
    y_pred = clf.predict(X_features)
    return clf, y_pred


def predict_classifier(model, clf, X):
    X_features = model.predict(X)
    return clf.predict(X_features)

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

mnist_train = (mnist.train.images > 0).reshape(55000, 28, 28, 1).astype(np.uint8) * 255
mnist_train = np.concatenate([mnist_train, mnist_train, mnist_train], 3)
mnist_test = (mnist.test.images > 0).reshape(10000, 28, 28, 1).astype(np.uint8) * 255
mnist_test = np.concatenate([mnist_test, mnist_test, mnist_test], 3)
print(mnist_train.shape)
# Load MNIST-M
mnistm = pkl.load(open('mnistm_data.pkl',"rb"))
mnistm_train = mnistm['train']
mnistm_test = mnistm['test']
mnistm_valid = mnistm['valid']
print(mnistm_train.shape)
print(mnistm_test.shape)
num_train =27500
num_test = 5000
combined_train_imgs = np.vstack([mnist_train[:num_train], mnistm_train[:num_train]])
combined_train_labels = np.vstack([mnist.train.labels[:num_train], mnist.train.labels[:num_train]])

combined_test_imgs = np.vstack([mnist_test[:num_test], mnistm_test[:num_test]])
combined_test_labels = np.vstack([mnist.test.labels[:num_test], mnist.test.labels[:num_test]])

#produce a domain class labels
# to train class labels
mnist_train_domain_class = np.zeros((27500,10) , dtype = int) # set zero for source domain
mnistm_train_domain_class = np.ones((27500,10), dtype = int) # set one for target domain
combined_train_domain_class = np.vstack([mnist_train_domain_class, mnistm_train_domain_class])
# test class  for testing domain classcifier
mnist_test_domain_class = np.zeros((5000,10), dtype = int)
mnistm_test_domain_class = np.ones((5000,10), dtype =int)
combined_test_domain_class = np.vstack([mnist_test_domain_class, mnistm_test_domain_class])



model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=[28, 28, 3]))
# model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(10, activation='relu'))
#model.summary()

model.compile(loss="mse", optimizer="adam")
print("compilation is over")

nnw = NNWeightHelper(model)
weights = nnw.get_weights()

def domain_classcifier():
	print("Total number of weights to evolve is:", weights.shape)

	all_examples_indices = list(range(combined_train_imgs.shape[0]))

	clf, _ = train_classifier(model, combined_train_imgs, combined_train_domain_class)
	print("combined_train_imgs shape is :", combined_train_imgs.shape)
	print("combined_train_labels shape is :",combined_train_domain_class.shape)
	y_pred = predict_classifier(model, clf, combined_test_imgs)
	print(combined_test_domain_class.shape, y_pred.shape)
	test_accuracy = accuracy_score(combined_test_domain_class, y_pred)

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
			clf, _ = train_classifier(model, combined_train_imgs[subsample_indices], combined_train_domain_class[subsample_indices])

			# calculate the predictions on a different set
			y_pred = predict_classifier(model, clf, combined_train_imgs[subsample_indices_valid])
			score = accuracy_score(combined_train_domain_class[subsample_indices_valid], y_pred)

			#clf, _ = train_classifier(model, combined_train_imgs, combined_train_domain_class)
			#y_pred = predict_classifier(model, clf, combined_test_imgs)
			#score = accuracy_score(combined_test_domain_class, y_pred)
			# append to array of values that are to be returned
			told.append(score)

		snes.tell(asked, told)
		end = timer()
		print("It took", end - start, "seconds to complete generation", i + 1)

	nnw.set_weights(snes.center)

	clf, _ = train_classifier(model, combined_train_imgs,combined_train_domain_class)
	y_pred = predict_classifier(model, clf, combined_test_imgs)

	print(combined_test_domain_class.shape, y_pred.shape)
	test_accuracy = accuracy_score(combined_test_domain_class, y_pred)
	print('Test accuracy:', test_accuracy)
	
# this function is to classcifier differ class both of mnist dataset and mnistm dataset  	
def class_classcifier():
	print("Total number of weights to evolve is:", weights.shape)

	all_examples_indices = list(range(combined_train_imgs.shape[0]))

	clf, _ = train_classifier(model, combined_train_imgs, combined_train_labels)
	print("combined_train_imgs shape is :", combined_train_imgs.shape)
	print("combined_train_labels shape is :",combined_train_labels.shape)
	y_pred = predict_classifier(model, clf, combined_test_imgs)
	print(combined_test_labels.shape, y_pred.shape)
	test_accuracy = accuracy_score(combined_test_labels, y_pred)
	
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
			clf, _ = train_classifier(model, combined_train_imgs[subsample_indices], combined_train_labels[subsample_indices])

			# calculate the predictions on a different set
			y_pred = predict_classifier(model, clf, combined_train_imgs[subsample_indices_valid])
			score = accuracy_score(combined_train_labels[subsample_indices_valid], y_pred)

			# clf, _ = train_classifier(model, x_train, y_train)
			# y_pred = predict_classifier(model, clf, x_test)
			# score = accuracy_score(y_test, y_pred)
			# append to array of values that are to be returned
			told.append(score)

		snes.tell(asked, told)
		end = timer()
		print("It took", end - start, "seconds to complete generation", i + 1)

	nnw.set_weights(snes.center)

	clf, _ = train_classifier(model, combined_train_imgs,combined_train_labels)
	y_pred = predict_classifier(model, clf, combined_test_imgs)

	print(combined_test_labels.shape, y_pred.shape)
	test_accuracy = accuracy_score(combined_test_labels, y_pred)

	print('Test accuracy:', test_accuracy)
	plt.plot(told)
	plt.title("loss model")
	plt.xlabel("genaration")
	plt.ylabel("loss_value")
	plt.legend("test",loc = "upper left")
	plt.show()
	


if __name__ == '__main__':
	#domain_classcifier()
	class_classcifier()
