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
from keras import backend as K
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from keras_helper import NNWeightHelper
from snes import SNES

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

batch_size = 128
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28

for i in range(0,3):
	mnistm = pkl.load(open('mnistm_data.pkl', 'rb'))
	global mnistm_train
	global mnistm_test
	global mnistm_valid
	mnistm_train = mnistm['train']
	mnistm_test = mnistm['test']
	mnistm_valid = mnistm['valid']
	mnistm_train = mnistm_train[ :, :, : , i]
	mnistm_valid = mnistm_valid[ :, :, : , i]
	mnistm_test = mnistm_test[ : , : , :, i]

	if K.image_data_format() == 'channels_first':
		mnistm_train = mnistm_train.reshape(mnistm_train.shape[0], 1, img_rows, img_cols)
		mnistm_test = mnistm_test.reshape(mnistm_test.shape[0], 1, img_rows, img_cols)
		mnistm_valid = mnistm_valid.reshape(mnistm_valid.shape[0], 1, img_rows, img_cols)
		input_shape = (1, img_rows, img_cols)
		
	else:
		mnistm_train = mnistm_train.reshape(mnistm_train.shape[0], img_rows, img_cols, 1)
		mnistm_test = mnistm_test.reshape(mnistm_test.shape[0], img_rows, img_cols, 1)
		mnistm_valid = mnistm_valid.reshape(mnistm_valid.shape[0], img_rows, img_cols,1)
		input_shape = (img_rows, img_cols, 1)

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
	model.add(Conv2D(64, kernel_size=(3, 3),
                 	activation='relu',
                 	input_shape=input_shape))
#model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))
	model.add(Flatten())
	model.add(Dense(6, activation='relu'))
	model.add(Dropout(0.5))

#model.add(Dense(num_classes, activation='softmax'))
model.summary()
model.compile(loss="mse",
              optimizer="adam",
              metrics=['accuracy'])

print("compilation is over")
#mnistm_train = mnistm_train[:, :, :, 0]
#mnistm_test = mnistm_test[:, :, :, 0]
#mnistm_valid = mnistm_valid[:, :, :, 0]
print(mnistm_train.shape)
nnw = NNWeightHelper(model)
weights = nnw.get_weights()

def main():
    print("Total number of weights to evolve is:", weights.shape)

    all_examples_indices = list(range(mnistm_train.shape[0]))
    
    mnistm_valid1 = mnistm_valid.reshape(5000*28*28,)
    print(mnistm_valid1.shape)

    clf, _ = train_classifier(model, mnistm_train,mnistm_valid1)

    y_pred = predict_classifier(model, clf, mnistm_test)
    print(mnistm_test.shape, y_pred.shape)
    
    test_accuracy = accuracy_score(mnistm_test, y_pred)

    print('Non-trained NN Test accuracy:', test_accuracy)
    # print('Test MSE:', test_mse)

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

        # iterate over the population
        for asked_j in asked:
            # set nn weights
            nnw.set_weights(asked_j)
            # train the classifer and get back the predictions on the training data
            clf, _ = train_classifier(model, mnistm_train[subsample_indices], mnistm_valid[subsample_indices])

            # calculate the predictions on a different set
            y_pred = predict_classifier(model, clf, mnistm_train[subsample_indices_valid])
            score = accuracy_score(mnistm_train[subsample_indices_valid], y_pred)

            # clf, _ = train_classifier(model, x_train, y_train)
            # y_pred = predict_classifier(model, clf, x_test)
            # score = accuracy_score(y_test, y_pred)
            # append to array of values that are to be returned
            told.append(score)

        snes.tell(asked, told)
        end = timer()
        print("It took", end - start, "seconds to complete generation", i + 1)

    nnw.set_weights(snes.center)

    clf, _ = train_classifier(model, mnistm_train, mnistm_valid)
    y_pred = predict_classifier(model, clf, mnistm_test)

    print(mnistm_test.shape, y_pred.shape)
    test_accuracy = accuracy_score(mnistm_test, y_pred)

    print('Test accuracy:', test_accuracy)


if __name__ == '__main__':
    main()
