# **Behavioral Cloning** 

## Udacity Cloning behaviour writeup

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

---

In comparison to course material the key code modificatoons / customizations are in:
* model.py containing neural net model is written from scrach but based on code and hints shown in "Project: Behaviour Cloning" section of the course
* modified generator to do augmentations and append side camera images
* tweaking parameters / network architecture

**Summary:** The final network was trained/validated/tested by using the images I recorded on my side. I didn't use any joystick for recording, and I'm not a quiet good video game player. 
The network architecture is based on Nvidia network as suggested in the course material. 
I did not use any transfer learning. In order to achieve satisfactory performance meeting assignement objectives I mostly focused on manipulating input data: added normalization, appending side camera images with modified steeting angle. The trained model ws tested only on the first track of the simulator. 

---

### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following **key files**:
* model.py - containing the script to create and train the model
* drive.py - script for driving the car in autonomous mode (umodified)
* model.h5 - containing a trained convolution neural network 
* README.md - writeup on project coding, challenges and possible improvements

Additional files:
* original_README.md - original project readme file with the description of problem to solve
* video.mp4 - one lap around the test track video; recordeed of simulator in autonomous mode using the trained model.h5 (captured with drive.py)


#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. 

The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

**Note:** Due to simplicy of Keras by large the code is similar to what was demonstrated in the Project notes. The main additions are in the input data manipulation and network parameters. I also organized the code into functions to improve its readability and allow faster testing of different blocks.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

Implemented model uses almost unaltered nVidia network. Therefore the network includes:
* 2 data pre-processing layers : normalization and cropping stage
* 5 convolutional layers (each followed by RELU)
* 4 dense layers (proceeded by flatten)

The network architecture is implemented in lines 83-118 of model.py. Additionally, a simple model is implemented in lines 43-49, and LeNet model is implemented lines 51-81. 


#### 2. Attempts to reduce overfitting in the model

First test of my model produced some good results. Thus, when running autonomous mode on this first track with the final version of trained model the car always stayed on the road.

#### 3. Model parameter tuning

For model parameters I did not make any major modifications. Adam optimizer, learning rate, batch size and epoch count are similar as in the project preparation guidelines.

#### 4. Appropriate training data

The bulk of modifications pertains to training data feed into the algorithm

Following the recommandations provided into the project preparation, I recorded data with the vehicle driving centered on the road, on the left side, on the right side. I also drive in the opposite way. 


### Model Architecture and Training Documentation

#### 1. Solution Design Approach

As mentioned previously, my approach follow what was advised in the project preparation instructions. 
Here are the differents steps I followed : 
* **Implemented simple model** - result at best car went til first sharp turn (before the bridge) and at the end of it fell off the road.
* **Implemented LeNet model** - car went over the bridge and fell into the water at first right turn. 
* **Added augmentation data - flipping and right/left cameras** - car went a little more away, but still not able to do a turn. 
* **Implemented Nvidia model** - one lap is done ! 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 83-118) consisted of exactly the same as Nvidia model. 

#### 3. Creation of the Training Set & Training Process

As explained, I followed the indication given in the project preparation and record on my own my data. I made three laps by driving centered in lane, one lap on right side, one lap on left side. 
Finally, I made the same recording but on the other way. 

As required by the assignement I crearted the final video with drive.py script showing the road from drivers perspective (see below picture).

[Youtube link to the video](https://youtu.be/p2exgPC5LGg)



