from keras.models import model_from_json
from pathlib import Path
from keras.preprocessing import image
import numpy as np
from keras.applications import vgg16

#load the json file that contains the model's structure
f = Path("model_structure.json")
model_structure = f.read_text()

#recreate the keras model object from the json data
model = model_from_json(model_structure)

#reload the model's trained weights
model.load_weights("model_weights.h5")

#load an image file to test, resizing it to 64X64 pixels (as model's required that)
img = image.load_img("dog.jpg", target_size=(64, 64))

#convert the image to a numpy array
image_array = image.img_to_array(img)

#add a fourth dimension to the image
images = np.expand_dims(image_array, axis=0)

#normalize the data
images = vgg16.preprocess_input(images)

#use the pretrained neural network to extract features from our test image
feature_extraction_model = vgg16.VGG16(weights="imagenet",
                                       include_top=False,
                                       input_shape=(64,64,3))
features = feature_extraction_model.predict(images)

#given the extracted features, make a final prediction using our own model
results = model.predict(features)

# Since we are only testing one image with possible class, we only need to check the first result's first element
single_result = results[0][0]

# Print the result
print("Likelihood that this image contains a dog: {}%".format(int(single_result * 100)))