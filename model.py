#!/usr/bin/env python

import csv
from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt


from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D


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
	model.add(Conv2D(filters = 6, kernel_size = 5, strides = 1, activation = 'relu'))

	# Pooling layer S2
	model.add(MaxPooling2D())

	# Convolution C3
	model.add(Conv2D(filters = 6, kernel_size = 5, strides = 1, activation = 'relu'))

	# Pooling layer S4
	model.add(MaxPooling2D())

	# Fully connected
	model.add(Flatten())

	model.add(Dense(120))

	model.add(Dense(84))

	model.add(Dense(1))
	
	return model

def augment_data(images, measurements):
	augmented_images, augmented_measurements = [], [] 

	for image, measurement in zip(images, measurements): 
		augmented_images.append(image)
		augmented_measurements.append(measurements)
		augmented_images.append(np.fliplr(image))
		augmented_measurements.append(-measurement)

	return np.array(augmented_images), np.array(augmented_measurements)


def use_left_right_camera(data_directory): 

	with open(data_directory+'driving_log.csv', 'r') as f:
		reader = csv.reader(f)
		for row in reader:
			measurement_center = float(row[3])

			# create adjusted steering measurements for the side camera images
			correction = 0.2 # this is a parameter to tune
			measurement_left = measurement_center + correction
			measurement_right = measurement_center - correction

			# read in images from center, left and right cameras
			source_path = row[0]
			filename = source_path.split('/')[-1]
			current_path = data_directory+'IMG'+filename
			img_center = plt.imread(current_path + row[0])
			img_left = plt.imread(current_path + row[1])
			img_right = plt.imread(current_path + row[2])

			# add images and angles to data set
			images.extend(img_center, img_left, img_right)
			measurements.extend(steering_center, steering_left, steering_right)
	return images, measurements



X_train, y_train = use_left_right_camera('./data/')
print(len(X_train))
X_train, y_train = augment_images(X_train, y_train)
print(len(X_train))

model = LeNet_model()
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=2)
model.save('model.h5')

