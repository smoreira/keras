import flask
import pandas as pd
import tensorflow as tf
import keras
from keras.models import load_model

# Inicializar Web Server
app = flask.Flask(__name__)

def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    keras.backend.get_session().run(tf.local_variables_initializer())
    return auc

global graph
graph = tf.get_default_graph()
model = load_model('games.h5', custom_objects={'auc': auc})


@app.route('/predict/', methods=['GET', 'POST'])
def predict():

    data = {"success": False}

    params = flask.request.json
    if (params == None):
        params = flask.request.args

    if (params != None):
        x=pd.DataFrame.from_dict(params, orient='index').transpose()
        with graph.as_default():
            data["prediction"] = str(model.predict(x)[0][0])
            data["success"] = True

    # retorna response em formato json
    return flask.jsonify(data)

# rodar o servico no ip e porta abaixo
app.run(host='0.0.0.0', port=5001)


#http://54.227.110.43:5000/predict?g1=1&g2=0&g3=0&g4=0&g5=0&g6=0&g7=0&g8=0&g9=0&g10=0