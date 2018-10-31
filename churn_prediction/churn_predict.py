
from keras.models import model_from_json
from keras.utils import np_utils
import numpy as np



# Model reconstruction from JSON file
with open('model.json', 'r') as f:
    model = model_from_json(f.read())

# Load weights into the new model
model.load_weights('model.h5')

Xnew = np.array([[608,2,1,41,1,83807.86,1,0,1,112542.58]]) #0
#Xnew = np.array([[622,2,1,46,4,107073.27,2,1,1,30984.59]]) #1
#Xnew = np.array([[559,0,0,49,2,147069.78,1,1,0,120540.83]]) #1

#print(Xnew.shape)

# make a prediction
#prediction = model.predict(Xnew) #inteiro
prediction = model.predict_classes(Xnew) #array

# show the inputs and predicted outputs
print("X=%s, Predicted=%s" % (Xnew[0], prediction[0]))