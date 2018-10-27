# Dog-Breed-Classification
Classifying 120 dog breeds using deep learning model
### Dataset description
Dataset is downloaded from Kaggle's Dog Breed Identification competition which comprises of training set and a test set of images of dogs. Each image has a filename that is its unique id. The dataset comprises 120 breeds of dogs. The goal  is to create a classifier capable of determining a dog's breed from a photo.
### Methodolgy
1. Preprocessing of data

    Execute preprocessing python code as 
   `
   python preprocessing.py
 ` 
It will preprocess Kaggle's given training dataset into two seperate folders of train and validation (0.3 of Training Dataset). Each folder will have seperate folders highlighting dog's breed and it's corresponding images.

2.Training of Model
Dog breed dataset is made using Imagent datset. Hence, Transfer learning is used for identifying dog breed from images. The model uses the pre-trained VGG-19 and Resnet-50 models as a fixed feature extractor, where the last convolutional output of both networks is fed as input to another, second level model. At the end, I combined both models to achieve a small boost relative to what I achieved by using them separately. Here are few lines, that extract the features from the images:

We only add a global average pooling layer and a fully connected layer, where the latter contains one node for each dog category and is equipped with a softmax. Let’s extract the last convolutional output for both networks.
