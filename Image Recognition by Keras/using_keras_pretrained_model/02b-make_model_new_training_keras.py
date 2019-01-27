from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from pathlib import Path
import joblib

"""
the code is exactly the same like training any neural networks but with two small differences.
1- how to load our training data
"""
#load data set
x_train = joblib.load("x_train.dat")
y_train = joblib.load("y_train.dat")

"""
2- how we define our neural network, since we use VGG16 to extract features from our image, this neural network has
no convolutional layers. Instead it only has the final dense layers of the neural network 
"""
#create a model and add layers
model = Sequential()

model.add(Flatten(input_shape=x_train.shape[1:]))
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(1, activation="sigmoid"))

#compile our model
model.compile(loss="binary_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

#train the model
model.fit(x_train,
          y_train,
          epochs=10,
          shuffle=True)

#save neural network structure
model_structure = model.to_json()
f = Path("model_structure.json")
f.write_text(model_structure)

#save neural network's trained weights
model.save_weights("model_weights.h5")
