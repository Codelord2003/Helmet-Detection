# Helmet-Detection
Problem Statement:
Helmet Detection on Two-Wheeler drivers using Convolutional Neural Network (CNN)

Designing a Machine Learning model to support traffic police in enforcing helmet-wearing regulations for two-wheeler drivers, aiming to promote behavioral change and enhance road safety. The model will utilize computer vision technology to accurately detect helmet usage, ultimately reducing accident severity and fostering safer driving practices.
 
Dataset:
The dataset contains 764 images of 2 distinct classes for the objective of helmet detection. Bounding box annotations are provided in the PASCAL VOC format
The classes are:
•	With helmet
•	Without helmet
 
About The Code
Importing Libraries:
Import necessary libraries for file and directory operations (os), computer vision processing (cv2), numerical operations (numpy), and deep learning with TensorFlow (tensorflow).

Setting Data Directory:
Specify the directory where your dataset is located (data_directory).

Preparing Class Labels:
Obtain a list of class labels by listing subdirectories in the dataset directory.
Create a dictionary (class_to_label) to map class labels to numerical indices.

Image Data Generator:
Create an instance of the ImageDataGenerator class for data augmentation and normalization. Specify rescaling of pixel values to a range of [0, 1].
Define a validation split of 15%.
 
Data Generators:
Create separate data generators for training and validation sets using the flow_from_directory method.
Resize images to a consistent size (e.g., 256x256 pixels).
Set the batch size and class mode as 'categorical.'

Load and Augment Data:
Loop through batches of training and validation data using the data generators.
Extend lists (X_train, y_train, X_validation, y_validation) with batch images and labels.

Convert to NumPy Arrays:
Convert the lists to NumPy arrays for further processing.

Check Data Shapes:
Check the shapes of the training data arrays to ensure they match the expected dimensions.

Convolutional Layers:
CNNs start with convolutional layers that convolve (slide) filters or kernels across the input image to extract local patterns and features.Each filter is responsible for detecting specific features, such as edges, textures, or higher-level structures.
 
Convolutional layers capture spatial hierarchies of features by combining local patterns into more complex patterns.

Activation Functions:
After each convolutional operation, an activation function, ReLU (Rectified Linear Unit), is applied element-wise to introduce non-linearity.
ReLU helps the network learn complex relationships in the data by introducing non- linearities.

Pooling Layers:
Pooling layers, often MaxPooling or AveragePooling, follow convolutional layers to reduce spatial dimensions and computational complexity.
Pooling layers retain the most important information from the feature maps, making the network more robust to variations in scale and orientation.

Flatten Layer:
After several convolutional and pooling layers, the feature maps are flattened into a vector.
The Flatten layer transforms the spatial information into a one-dimensional vector, preparing it for input to fully connected layers. Fully Connected (Dense) Layers:
 
Fully connected layers take the flattened vector and perform a series of linear operations.
These layers learn global patterns and relationships in the data, combining features from different parts of the image.
The last fully connected layer often has a softmax activation function for multi-class classification, providing class probabilities.
Dropout:

Dropout layers are introduced to prevent overfitting.
During training, randomly selected neurons are ignored, reducing the reliance on specific features and promoting a more robust model.

Output Layer:
The output layer produces the final predictions based on the learned features.
For classification tasks, the sigmoid activation function is commonly used to convert the network's output into class probabilities.

Accuracy:
For Training: 90.17%
For Validation: 76.10%
