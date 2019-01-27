from pathlib import Path
import numpy as np
from keras.preprocessing import image
from keras.applications import vgg16
import joblib

#paths to folders with training data
dog_path = Path("training_data") / "dogs"
not_dogs_path = Path("training_data") / "not_dogs"

images = []
labels = []

#load all the not-dogs images
for img in not_dogs_path.glob("*.png"):
    #load the image from disk
    img = image.load_img(img)

    #convert the image to numpy array
    image_array = image.img_to_array(img)

    #add the image to the list of images
    images.append(image_array)

    #for each "not dog" image, the expected value should be zero
    labels.append(0)

#load all the dogs images
for img in dog_path.glob("*.png"):
    #load the image from disk
    img = image.load_img(img)

    #convert the image to numpy array
    image_array = image.img_to_array(img)

    #add the image to the list of images
    images.append(image_array)

    #for each "not dog" image, the expected value should be zero
    labels.append(1)

#create a single numpy array with all the images we loaded
x_train = np.array(images)

#also convert the labels into a numpy array
y_train = np.array(labels)

#normalize image data into 0-to-1 range
x_train = vgg16.preprocess_input(x_train)

#load a pre-trained neural network to use as a feature extractor
pretrained_nn = vgg16.VGG16(weights="imagenet", include_top=False, input_shape=(64, 64, 3))

#extract features for each image
feature_x = pretrained_nn.predict(x_train)

#save the array of extracted features to a file
joblib.dump(feature_x, "x_train.dat")

#save the matching array of expected value to a file
joblib.dump(y_train, "y_train.dat")