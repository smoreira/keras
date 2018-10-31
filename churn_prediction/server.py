from flask import Flask, render_template, request
from scipy.misc import imread, imresize
import numpy as np
import re
import sys
import os

sys.path.append(os.path.abspath("./"))
from load import *

# initalize our flask app
app = Flask(__name__)

# global vars for easy reusability
global model, graph

# initialize these variables
model, graph = init()

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict/', methods=['GET', 'POST'])
def predict():

    # read the image into memory
    #x = np.array([[608,2,1,41,1,83807.86,1,0,1,112542.58]])
    x = np.array([[622,2,1,46,4,107073.27,2,1,1,30984.59]])

    # in our computation graph
    with graph.as_default():
        # perform the prediction
        out = model.predict(x)
        print(out)
        print(np.argmax(out, axis=1))
        # convert the response to a string
        response = np.argmax(out, axis=1)
        return str(response[0])

if __name__ == "__main__":
    # run the app locally on the given port
    app.run(host='0.0.0.0', port=5000)
# optional if we want to run in debugging mode
# app.run(debug=True)