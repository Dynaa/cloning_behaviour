#!/usr/bin/env python

import csv
from scipy import ndimage
import numpy as np 
from keras.models import Sequential
from keras.layers import Flatten, Dense

from keras.layers import Lambda


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
	model.add(Lambda(lambda x: (x/ 255.0) - 0.5, input_shape=(160,320,3)))
	model.add(Flatten())
	model.add(Dense(1))

	return model 


X_train, y_train = read_data('./data/')

model = simple_model()
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=7)
model.save('model.h5')

