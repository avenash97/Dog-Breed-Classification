# Dog-Breed-Classification
Classifying 120 dog breeds using deep learning model
### Dataset description
Dataset is downloaded from Kaggle's Dog Breed Identification competition which comprises of training set and a test set of images of dogs. Each image has a filename that is its unique id. The dataset comprises 120 breeds of dogs. The goal  is to create a classifier capable of determining a dog's breed from a photo.
### Packages Required
```
1. keras
2. tensorflow
3. matplotlib
4. sklearn
5. pillow
6. numpy
7. pandas
```

### Methodolgy
1. Preprocessing of data

    Execute preprocessing python code as 
   `
   $ python preprocessing.py
 ` 
It will preprocess Kaggle's given training dataset into two seperate folders of train and validation (0.3 of Training Dataset). Each folder will have seperate folders highlighting dog's breed and it's corresponding images.

2. Training of Model
* Dog breed dataset is made using Imagenet datset. Transfer learning on models which are trained on Imagenet dataset will be efficient for identifying dog breed from images. 
* The model uses the pre-trained VGG-19 and Resnet-50 models as a feature extractor, where the last convolutional output of both networks is fed as input to another, second level model comprising of dense layers, dropout and softmax classification layer.
* At the end, both models were combined to achieve a small boost in the result compared to individually achieved by using them separately. 

3. Execute training python code as 
  
  ```
   $ python dog.py
  ```

`model.summary()` will fetch the below image
![alt text](https://github.com/avenash97/Dog-Breed-Classification/blob/master/Screenshot%20from%202018-10-27%2021-05-22.png)

### Results of the model

Model was trained for 200 Epochs and by running tensorboard for logs will fetch the below graphs.
   
   ```
  $ tensorboard --logdir=/logs
  ```
![alt text](https://github.com/avenash97/Dog-Breed-Classification/blob/master/Images/Accuracy1.png)

![alt text](https://github.com/avenash97/Dog-Breed-Classification/blob/master/Images/Loss1.png)



* Accuracy - 97.85 % was achieved and can be verified in above plot of accuracy vs epochs.
* Loss - 0.1171 was achieved and can be verified in above plot of loss vs epochs
