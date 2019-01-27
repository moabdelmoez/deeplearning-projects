import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

from pathlib import Path
import matplotlib.pyplot as plt

#list the names for each CIFAR10 class
cifar10_class_names = {
    0: "Plane",
    1: "Car",
    2: "Bird",
    3: "Cat",
    4: "Deer",
    5: "Dog",
    6: "Frog",
    7: "Horse",
    8: "Boat",
    9: "Truck"
}

#load the entire dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

#loop through each image in this range to display the images
for i in range(10):
    #grab an image from the dataset
    sample_image = x_train[i]
    #grab the image's expected class id
    image_class_number = y_train[i][0]
    #look up the class name from the class id
    image_class_name = cifar10_class_names[image_class_number]

    #draw the image as a plot
    plt.imshow(sample_image)
    #label the image
    plt.title(image_class_name)
    #show the plot on the screen
    plt.show()

#normalize dataset to 0-to-1 range
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = x_train/255
x_test = x_test/255

"""convert class vectors to binary class metrix, our labels are single values 
from 0 to 9. Instead we want each lable to be an array with one element equal to 1
and the others are 0
"""
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

#create a Sequential model and add layers
model = Sequential()

#add convolutional layers
model.add(Conv2D(filters=32, kernel_size=(3,3), padding="same",activation="relu",
                 input_shape=(32,32,3)))
model.add(Conv2D(filters=32, kernel_size=(3,3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=64, kernel_size=(3,3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

#flatten the model
model.add(Flatten())

model.add(Dense(512, activation="relu"))
model.add(Dropout(0.50))
model.add(Dense(10, activation="softmax"))

#complie the model
model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

#Print a summaryof the model
# model.summary()

#train the model
"""batch_size is how many images we want to feed into the network at once during
training. Typical batch_size are between 32 and 128
epochs is we need to decide how many times we wanna go through our training data set
during the process. One full pass through the entire training data set 
is called an epoch.
Next, we need to tell Keras what data we wanna use to validate our training.
This is data that the model will never see during training, 
and it'll only be used to test the accuracy of the training model.
shuffle means to randomize the training data order
"""
model.fit(x_train,
          y_train,
          batch_size=64,
          epochs=10,
          validation_data=(x_test, y_test),
          shuffle=True)

#save neural network structure

#convert the model structure to json format
model_structure = model.to_json()
#create and write the the json structure to a file
f = Path("model_structure.json")
f.write_text(model_structure)

#save neural network's trained weights
model.save_weights("model_weights.h5")