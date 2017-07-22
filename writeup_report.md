**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/model.png "Model Architecture"
[image2]: ./examples/nvidia_cnn.png "Model Visualization"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

I used two models. One is small_model which is easy to use locally for experiments and to check the pipeline and nvidia model which is recommended by lections which was trained in AWS. Both models have several convolution layers and include RELU layers to introduce nonlinearity.
The data is normalized in models using a Keras lambda layer.

####2. Attempts to reduce overfitting in the model

In order to reduce overfitting I used more data.

The final model was trained and validated on different data sets to ensure that the model was not overfitting and number of epochs was adjusted properly. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, driving counter-clockwise, flipping the images, data from the second track, data from multiple cameras.

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to use simple model in order to check whole pipeline on laptop and train the model in order to drive first track and then use nvidia model and more data in order to pass second track. 

According to "CS231n Winter 2016: Lecture 11: ConvNets in practice" there is no sense to use big filters, so for the simple model I decided to simplify nvidia model by replacing all 5x5 filters by 3x3 filters and reduce number of layers until it will have reasonable performance on laptop. By doing that, I was able to train small simplified model on data provided by udacity in order to drive first track. It was a good start and good tool for experiments, but such model was not able to drive second track.

After that I collected more training data and trained nvidia model in AWS. In order to avoid overfitting, I checked  MSE on the training data and validation data and set appropriate number of epochs to prevent overfitting.

Using nvidia model, I noticed that car can drive first track easily, even trained only on provided by Udacity training data, however, still can not pass second track. I spent long hours training nvidia model in AWS trying to use different offsets for images from left and right cameras, don't using images from left and right cameras at all and so on. But still without any big progress.

After that I noticed that car tend to drive in forward direction. I checked steering angles distribution and it turned out that I had a lot of training samples with steering angle close to zero. After that I decided to filter out some of them and make distribution of steering angles close to uniform destribution.

I did so, but now car was able to drive second track and had a problem with first track that it started to wag on road. Even though, the car still stayed on the road but I was not satisfied with such behavior.

I investigated different techniques which can be used in order to force the car to converge in forward direction in such cases, but decided to take simpler approach and just take a little bit more training data which have steering angle close to zero, hoping that model will learn to prefer forward direction, but not too much forward direction. And eventually, it helped.

Apart from that, I noticed that it seems that performance of autonomous mode depends on quality of display mode, it seems in certain cases if quality is too high drive.py can not catch up, and behavior is not good.

At the end of the process, the vehicle is able to drive autonomously around both tracks without leaving the road.

####2. Final Model Architecture

I used the model described by [nVidia paper](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)

Here is a visualization of the architecture.

![alt text][image2]

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover back to center.
These images show what a recovery looks like starting from right side :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would help the model generalize. For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

After the collection process, I had 144027 number of data points. I then preprocessed this data by filter out some points in order to make data more uniform and had got 65048 number of data points.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 4 as evidenced by MSE on validation set, since MSE started to grow from 5th epoch. I used an adam optimizer so that manually training the learning rate wasn't necessary.
