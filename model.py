#!/usr/bin/env python

import csv
from scipy import ndimage
import numpy as np 
from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Lambda, Cropping2D


def read_data(data_directory): 
	lines = []
	with open(data_directory+'driving_log.csv') as csvfile:
		reader = csv.reader(csvfile)
		next(reader, None)  # skip the headers #skip header
		for line in reader:
			lines.append(line)

	print(line)

	images = []
	measurements = []
	for line in lines:
		source_path = line[0]
		filename = source_path.split('/')[-1]
		current_path = data_directory+'IMG/' + filename
		#print(current_path)
		image = ndimage.imread(current_path)
		images.append(image)
		measurement = float(line[3])
		measurements.append(measurement)

	X_train = np.array(images)
	y_train = np.array(measurements)

	return X_train, y_train

def simple_model():
	model = Sequential()
	model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
	model.add(Flatten())
	model.add(Dense(1))

	return model 

def LeNet_model(): 
	model = Sequential()

	# Crop images 
	model.add(Cropping2D(cropping=((70,25),(0,0)), input_shape=(160,320,3)))

	# Normalization 
	model.add(Lambda(lambda x: (x/ 255.0) - 0.5))

	# Convolution C1 
	model.add(Convolution2D(filters = 6, kernel_size = 5, strides = 1, activation = 'relu'))

	# Pooling layer S2
	model.add(MaxPooling2D())

	# Convolution C3
	model.add(Convolution2D(filters = 6, kernel_size = 5, strides = 1, activation = 'relu'))

	# Pooling layer S4
	model.add(MaxPooling2D())

	# Fully connected
	model.add(Flatten())

	model.add(Dense(120))

	model.add(Dense(84))

	model.add(Dense(1))
	
	return model


X_train, y_train = read_data('./data/')

model = LeNet_model()
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=2)
model.save('model.h5')

