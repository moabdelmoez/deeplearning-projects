from keras.models import model_from_json
from pathlib import Path
from keras.preprocessing import image
import numpy as np

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

#load the json file that contains model's structure
f = Path("model_structure.json")
prebuilt_model_structure = f.read_text()

#recreate the keras model objects fron the json file
model = model_from_json(prebuilt_model_structure)

#reload the model's trained weights
model.load_weights("model_weights.h5")

#load an image file to test, and also resizing it to 32 x 32 pixels, as our model
img = image.load_img("car.jpg", target_size=(32, 32))
print(img)

#convert the image to a 3-D numpy array
image_to_test = image.img_to_array(img)

#add a fourth dimension to the image (since Keras expects a list of images, not a single image)
list_of_images = np.expand_dims(image_to_test, axis=0)

#make a prediction uding the model
results = model.predict(list_of_images)

#since we are only using one image, we need to just check the first result
single_result = results[0]

#we will get the most likelihood score for all 10 possible calsses
most_likely_class_index = int(np.argmax(single_result))
most_likelihood_class = single_result[most_likely_class_index]

#get the name of the most likely class
class_label = cifar10_class_names[most_likely_class_index]

print("This is image is a {} - Likelihood: {:2f}".format(class_label, most_likelihood_class))