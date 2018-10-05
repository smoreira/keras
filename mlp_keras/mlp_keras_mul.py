from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

#Dados
dataset = np.loadtxt("winequality-red.csv", delimiter=";")

# Separando colunas input (X) e output (Y) 
X = dataset[:,0:10]
labels = dataset[:,11]

#print(labels)

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(labels)
encoded_Y = encoder.transform(labels)
Y = np_utils.to_categorical(encoded_Y)

#print (Y)

#Modelo (fully-connected layers)
model = Sequential()
model.add(Dense(64, kernel_initializer='uniform', activation='relu', input_dim=10))
model.add(Dense(32, kernel_initializer='uniform', activation='relu'))
model.add(Dense(64, kernel_initializer='uniform', activation='relu'))
model.add(Dense(32, kernel_initializer='uniform', activation='relu'))
model.add(Dense(6, kernel_initializer='uniform', activation='sigmoid'))

#Compilando
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

#Treinando
model.fit(X, Y,
          epochs=600,
          batch_size=20)

#Avaliando
scores = model.evaluate(X, Y, batch_size=10)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


# Resumo da Rede
print(model.summary())