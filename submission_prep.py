from sklearn.datasets import load_files       
from keras.utils import np_utils
import numpy as np
from glob import glob
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input as preprocess_input_vgg19
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input as preprocess_input_resnet50
from keras.preprocessing import image                  
from tqdm import tqdm
from PIL import ImageFile                            
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers import Input, Dense
from keras.layers.core import Dropout, Activation
from keras.callbacks import ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.callbacks import TensorBoard
import time
import cv2
import pickle
def load_dataset(path):
    data = load_files(path)
    #print(data)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 120)
    return dog_files, dog_targets

train_files, train_targets = load_dataset('/home/avenash/dog_breed/data_gen/train')
valid_files, valid_targets = load_dataset('/home/avenash/dog_breed/data_gen/validation')
test_files, test_targets = load_dataset('/home/avenash/dog_breed/data_gen/test/test')
#print(sorted(glob("/home/avenash/dog_breed/data_gen/train/**/")))
dog_names = [item[20:-1] for item in sorted(glob("/home/avenash/dog_breed/data_gen/train/**/"))]
print('There are %d total dog categories.' % len(dog_names))
print('There are %s total dog images.\n' % len(np.hstack([train_files, valid_files, test_files])))
print('There are %d training dog images.' % len(train_files))
print('There are %d validation dog images.' % len(valid_files))
print('There are %d test dog images.'% len(test_files))


def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)
	
ImageFile.LOAD_TRUNCATED_IMAGES = True                 

# pre-process the data for Keras
#train_tensors = paths_to_tensor(train_files).astype('float32')/255
#valid_tensors = paths_to_tensor(valid_files).astype('float32')/255
test_files = sorted(glob("/home/avenash/dog_breed/data_gen/test/test/*.jpg"))
test_tensors = paths_to_tensor(test_files).astype('float32')/255


def extract_VGG19(file_paths):
    tensors = paths_to_tensor(file_paths).astype('float32')
    preprocessed_input = preprocess_input_vgg19(tensors)
    return VGG19(weights='imagenet', include_top=False).predict(preprocessed_input, batch_size=32)

def extract_Resnet50(file_paths):
    tensors = paths_to_tensor(file_paths).astype('float32')
    preprocessed_input = preprocess_input_resnet50(tensors)
    return ResNet50(weights='imagenet', include_top=False).predict(preprocessed_input, batch_size=32)

def extract_Resnet50_single(img):
    #tensors = paths_to_tensor(file_paths).astype('float32')
    preprocessed_input = preprocess_input_resnet50(np.expand_dims(img.astype('float32'),axis=0))
    return ResNet50(weights='imagenet', include_top=False).predict(preprocessed_input, batch_size=128)


def extract_VGG19_single(img):
    #tensors = paths_to_tensor(file_paths).astype('float32')
    preprocessed_input = preprocess_input_vgg19(np.expand_dims(img.astype('float32'),axis=0))
    return VGG19(weights='imagenet', include_top=False).predict(preprocessed_input, batch_size=128)
#train_vgg19 = extract_VGG19(train_files)
#valid_vgg19 = extract_VGG19(valid_files)
test_vgg19 = extract_VGG19(test_files)
#p''rint("VGG19 shape", train_vgg19.shape[1:])

#train_resnet50 = extract_Resnet50(train_files)
#valid_resnet50 = extract_Resnet50(valid_files)
test_resnet50 = extract_Resnet50(test_files)
#print("Resnet50 shape", train_resnet50.shape[1:])


def input_branch(input_shape=None):
    
    size = int(input_shape[2] / 4)
    
    branch_input = Input(shape=input_shape)
    branch = GlobalAveragePooling2D()(branch_input)
    branch = Dense(size, use_bias=False, kernel_initializer='uniform')(branch)
    branch = BatchNormalization()(branch)
    branch = Activation("relu")(branch)
    return branch, branch_input

vgg19_branch, vgg19_input = input_branch(input_shape=(7, 7, 512))
resnet50_branch, resnet50_input = input_branch(input_shape=(1, 1, 2048))
concatenate_branches = Concatenate()([vgg19_branch, resnet50_branch])

net = Dropout(0.3)(concatenate_branches)
net = Dense(640, use_bias=False, kernel_initializer='uniform')(net)
net = BatchNormalization()(net)
net = Activation("relu")(net)
net = Dropout(0.3)(net)
net = Dense(120, kernel_initializer='uniform', activation="softmax")(net)

model = Model(inputs=[vgg19_input, resnet50_input], outputs=[net])
model.summary()

model.compile(loss='categorical_crossentropy', optimizer="rmsprop", metrics=['accuracy'])
checkpointer = ModelCheckpoint(filepath='saved_models/bestmodel.hdf5', verbose=1, save_best_only=True)

#keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch')
tensorboard = TensorBoard(log_dir="logs/")
#model.fit([train_vgg19, train_resnet50], train_targets, validation_data=([valid_vgg19, valid_resnet50], valid_targets),epochs=200, batch_size=8, callbacks=[checkpointer,tensorboard], verbose=1)
		  
model.load_weights('saved_models/bestmodel.hdf5')

from sklearn.metrics import accuracy_score

output={}
c = 0
'''
for img in glob("/home/avenash/dog_breed/data_gen/test/test/*.jpg"):
        name = img
        img  = cv2.imread(img)
        #print (img.shape)
        img = cv2.resize(img,(224,224))
        vgg_feature = extract_VGG19_single(img)
        resnet_feature = extract_Resnet50_single(img)
        predictions = model.predict([vgg_feature,resnet_feature])
        output[name] = predictions
        print(c)
        c+=1
pickle.dump(output,open("output.pkl","wb"))
        #print(predictions)
'''
prediction = model.predict([test_vgg19, test_resnet50])
print(prediction.shape)
pickle.dump(prediction,open("output.pkl","wb"))
#breed_predictions = [np.argmax(prediction) for prediction in predictions]
#breed_true_labels = [np.argmax(true_label) for true_label in test_targets]
#print('Test accuracy: %.4f%%' % (accuracy_score(breed_true_labels, breed_predictions) * 100))

