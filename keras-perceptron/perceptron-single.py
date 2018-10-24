from keras.datasets import mnist #data
from keras.models import Sequential #each layer feeds into next layer
from keras.layers import Dense, Flatten, Dropout #NN is a series of layers, we grabbed some
from keras.utils import np_utils

import wandb
from wandb.keras import WandbCallback

# logging code
run = wandb.init()
config = run.config

# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

#Convert our training sets into an array of 1 for 5 and 0 for not 5, uses numpy
is_five_train = y_train == 5
is_five_test = y_test == 5
labels = ["Not Five", "Is Five"]

img_width = X_train.shape[1]
img_height = X_train.shape[2]

# create model
model=Sequential()
model.add(Flatten(input_shape=(img_width,img_height))) #2D to 1D array
model.add(Dense(1)) #We have 1 perceptron, densely connected. Every input from old layer goes into 1 perceptron (how many outputs)
#model.add(Dense(1,activation="sigmoid")) #Map output between 0 and 1, this is our activation func
#how do we measure how far we stray from the truth, mse = mean squared error.
#Need to minimize loss
model.compile(loss='mse', optimizer='adam',
                metrics=['accuracy'])

# Fit the model
model.fit(X_train, is_five_train, epochs=3, validation_data=(X_test, is_five_test),
                    callbacks=[WandbCallback(data_type="image", labels=labels, save_model=False)])
model.save('perceptron.h5')
