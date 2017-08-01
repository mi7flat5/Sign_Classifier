#**Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./nopass.png "Traffic Sign 1"
[image5]: ./novh.png "Traffic Sign 2"
[image6]: ./stop.png "Traffic Sign 3"
[image7]: ./wild.png "Traffic Sign 4"
[image8]: ./60.png "Traffic Sign 5"

## Rubric Points
---

#Here is a link to my [project code](https://github.com/mi7flat/Sign_Classifier/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used vanilla python to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43



###Design and Test a Model Architecture

####1. Data preprocessing

With the Lenet code I was already getting high validation so I chose only to normalize the data set. I tried both grayscaling and converting to HSV with great results for test, validation and training sets, but classifying images from the web were terrible with this method, so I chose to keep it simple and am very pleased with the results. I normalized the pictures to values between -1 and 1. 




####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution      	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 3x3	    | 1x1 stride Valid Padding outputs 10x10x128     									|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  Valid Padding 		outputs 5x5x128 		|
| Convolution	    | 1x1 stride Valid Padding outputs 1x1x512       									|
| RELU					|												|
| Max pooling	      	| 3x3 stride,  Same padding		outputs 1x1x512		|
| Fully connected		|  Input 512  Outout 600      									|
| Droput | keep probability 0.5 |
| RELU					|												|
| Fully connected		|   Input 600 Output 230   									|
| Droput | keep probability 0.5 |
| RELU					|												|
| Output Layer |  Input 230 output 43 

 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I used the batch size that LeNet used as it didn't seem to make a difference in anything other than processing times if I increased it. I chose to do 20 epochs for this model because this would insure that my model reached a validation set accuracy of 93 percent though it often reaches that much sooner. I used a less agressive learning rate thatn LeNet because it would often get stuck bouncing back in forth in arbitrary value ranges. 

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93.

I chose to stay close to LeNet architecture, though in order to reach 93% validation accuracy I had to make a few adjustements. 
The validation and training accuracy calculations were calculated in an evaluate function called at the end of each epoch in training. 

In oder to reach higher validation accuracy I used dropout on the two fully connected layers. I also added an extra convolutional layer. 
I origionally added the third convolution layer so that I could use dropout on it, but I got better results with the layer not using dropout. I found that changing the size of layer outputs had a strong effect on validation accuracy, so I experimented with very large and resonably small numbers in each layer to see what the results were, and this is how i arrived at my final architecture. 

My final model results were:
* training set accuracy of .996
* validation set accuracy of .938
* test set accuracy of .929


###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| Probability
|:---------------------:|:---------------------------------------------:| 
| No Passing      		| No Passing   									| 1.0|
| Stop     			| Stop 										| 1.0 |
| No Vehicles					| No Vehicles										| .97|
| Roundabout Mandatory | Roundabout Mandatory | .99 |
|  Wild Animals  		| Wild Animals					 				| 1.0
| 60 km/h				| 60 km/h	      							| 1.0

I had the model predict each sign 5 times and the model performs on them about as well as it does on the training, validation and test sets. Calculating this a coule times I generally come up between 70% and 90% accuracy. I chose each picture and cropped it to a 32x32 image as the network expects. I do think the network is probably brittle in reguards to translated, rotated or skewed pictures, as it has not been trained under those conditions, but I don't think it is far from being useful. It would be a good part of a larger perception pipeline. 






