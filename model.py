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
	images = []
	measurements = []
	for line in lines:
		source_path = line[0]
		filename = source_path.split('/')[-1]
		#print(filename)
		current_path = data_directory+'IMG/' + filename
		#print(current_path)
		image = ndimage.imread(current_path)
		images.append(image)
		measurement = float(line[3])
		measurements.append(measurement)

	X_train = np.array(images)
	y_train = np.array(measurements)

	return X_train, y_train

def read_csv_file(data_directory):
	return 0

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

def Nvidia_model(): 
	model = Sequential()

	# Normalization 
	model.add(Lambda(lambda x: (x/ 255.0) - 0.5, input_shape=(160,320,3)))
    
    # Crop images 
	model.add(Cropping2D(cropping=((70,25),(0,0))))

	# Convolution 1
	model.add(Convolution2D(24,5,5,subsample=(2,2), activation = 'relu'))

	# Convolution 2
	model.add(Convolution2D(36,5,5,subsample=(2,2), activation = 'relu'))

	# Convolution 3
	model.add(Convolution2D(48,5,5,subsample=(2,2), activation = 'relu'))

	# Convolution 4
	model.add(Convolution2D(64,3,3, activation = 'relu'))

	# Convolution 5
	model.add(Convolution2D(64,3,3, activation = 'relu'))

	# Fully connected
	model.add(Flatten())

	model.add(Dense(100))

	model.add(Dense(50))

	model.add(Dense(10))
    
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
		steering_center = float(line[3])

		# create adjusted steering measurements for the side camera images
		correction = 0.2 # this is a parameter to tune
		steering_left = steering_center + correction
		steering_right = steering_center - correction

		# read in images from center, left and right cameras
		filename_center = line[0].split('/')[-1]
		filename_left = line[1].split('/')[-1]
		filename_right = line[2].split('/')[-1]
		current_path_center = data_directory+'IMG/'+filename_center
		current_path_left = data_directory+'IMG/'+filename_left
		current_path_right = data_directory+'IMG/'+filename_right
		img_center = ndimage.imread(current_path_center)
		img_left = ndimage.imread(current_path_left)
		img_right = ndimage.imread(current_path_right)

		# add images and angles to data set
		images.append(img_center)
		images.append(img_left)
		images.append(img_right)
		measurements.append(steering_center)
		measurements.append(steering_left)
		measurements.append(steering_right)

	return np.array(images), np.array(measurements)


def create_sample(data_directory): 
	samples = []
	with open(data_directory+'./driving_log.csv') as csvfile:
		reader = csv.reader(csvfile)
		for line in reader:
			samples.append(line)

	return samples

def generator(samples, batch_size=32):
	num_samples = len(samples)
	while 1: # Loop forever so the generator never terminates
		shuffle(samples)
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset:offset+batch_size]

			images = []
			angles = []
			for batch_sample in batch_samples:
				
				# load center image
				name_center = './IMG/'+batch_sample[0].split('/')[-1]
				center_image = cv2.imread(name_center)
				images.append(center_image)

				# flip center image
				images.append(np.fliplr(center_image))

				# load left image
				name_left = './IMG/'+batch_sample[1].split('/')[-1]
				left_image = ndimage.imread(name_left)
				images.append(left_image)

				# flip left image
				images.append(np.fliplr(left_image))

				# load right image	
				right_name = './IMG/'+batch_sample[2].split('/')[-1]
				right_image = ndimage.imread(right_name)
				images.append(right_image)

				# flip right image
				images.append(np.fliplr(right_image))

				# load center angle
				center_angle = float(batch_sample[3])
				angles.append(center_angle)

				# flip center angle 
				angles.append(-center_angle)

				# compute left and right angles
				correction = 0.2 # this is a parameter to tune
				left_angle = center_angle + correction
				right_angle = center_angle - correction
				angles.append(left_angle)
				angles.append(right_angle)

				# flip left and right angle
				angles.append(-left_angle)
				angles.append(-right_angle)

            # trim image to only see section with road
			X_train = np.array(images)
			y_train = np.array(angles)
			yield sklearn.utils.shuffle(X_train, y_train)	


#X_train, y_train = read_data('./data/')
#X_train, y_train = use_left_right_camera('./data/')
#print(len(X_train))
#X_train, y_train = augment_data(X_train, y_train)
#print(len(X_train))

samples = create_sample('./data/')
train_samples, validation_samples = train_test_split(sample_list, test_size=0.2)

# Set our batch size
batch_size=32

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)


model = Nvidia_model()
model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, /
            steps_per_epoch=ceil(len(train_samples)/batch_size), /
            validation_data=validation_generator, /
            validation_steps=ceil(len(validation_samples)/batch_size), /
            epochs=5, verbose=1)
#model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=5)
model.save('model.h5')
print("End")

