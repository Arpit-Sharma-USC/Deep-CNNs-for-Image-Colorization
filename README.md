# Deep-CNNs-for-Image-Colorization

A convolutional neural network for image colorization which turns a grayscale image to a colored image. By converting an image to grayscale, we loose color information, so converting a grayscale image back to a colored version is not an easy job. I will use the CIFAR-10 dataset.

# Dataset
http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz

# Procedure

•	Data Filtering for birds:
o	To begin with I used the cifar10 load_data method. And then picked those whose among the whole data whose indices matched with that of birds.
•	Picking pixels:
o	 I decided to pick 40% of the pixels.
•	After running the k-means algorithm on the data at hand, I obtained the cluster centers
•	Image shape Transformation:
o	I used transform from skimage to reshape the image to 32 x 32 x 1. 
•	The CNN Model:
o	Approach:
•	At first, I used my PC to run 6 epochs and compared the results. I then ran it on a supercomputer with TESLA- K80 GPU for tensorflow and was able to run high number of Epochs for this test, out of curiosity.
•	I decided not to use the padding and used a stride of length two.
•	Input is of shape (32, 32) which is a Greyscale Image. 
•	Using the cluster centers obtained I recolored images(having only 4 colors) as the output and not the original CIFAR-10 images.
•	The output is expected to have a shape (1024, 4) where 1024 is the number of pixels and 4 is the one hot encoded value of that pixel. In other words, for instance pixel 1 has color 2, [0, 0, 1, 0]. 
•	Moving forward, the second layer in the network will have 4096 units, which will then be reshaped to (1024, 4) units using another Reshape layer. 
•	After this, I obtained a 2D grid in the network of the shape (1024, 4) (which matches the expected output shape). I the ndecided to add another layer, namely the Activation layer with softmax which helped to choose the highest value out of 4 inputs for each of the 1024 pixels. 
•	After receiving the 1024 output counts, I built a compilation for the model with loss='categorical_crossentropy'.

The validation set of chosen as 500 of the bird samples, training as 4500 and the test set had 1000 instances.
