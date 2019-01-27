import numpy as np
from keras.preprocessing import image
from keras.applications import vgg16

#load keras' VGG16 model that was pre-trained against the ImageNet database
model = vgg16.VGG16()

#load the image file, resizing it to 224X224 pixels (required by model)

img = image.load_img("sea.jpg", target_size=(224, 224))

#convert the image to a numpy array
x = image.img_to_array(img)

#add a fourth dimension
x = np.expand_dims(x , axis=0)

#normalize the input image's pixel values to the range when training neural network between 0 to 1
x = vgg16.preprocess_input(x)

#run the image through the deep neural network to make a prediction
predictions = model.predict(x)

#look up the names of the predicted class, and decode it
predicted_classes = vgg16.decode_predictions(predictions, top=9)

print("Top predictions for the image: ")

for imagenet_id, name, likelihood in predicted_classes[0]:
    print("Prediction: {} - {:2f}".format(name, likelihood))