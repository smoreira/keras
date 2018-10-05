# Criando a primeira rede com Keras - Classificação Binária

from keras.models import Sequential
from keras.layers import Dense
import numpy

# Random Seed
seed = 7
numpy.random.seed(seed)

# Carregando dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")

# Separando colunas input (X) e output (Y) 
X = dataset[:,0:8]
Y = dataset[:,8]

# Criando Modelo
model = Sequential()
model.add(Dense(64, input_dim=8, kernel_initializer='uniform', activation='relu'))
model.add(Dense(32, kernel_initializer='uniform', activation='relu'))
model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))

# Compilando Modelo
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Treinando Modelo
model.fit(X, Y, epochs=800, batch_size=15)

# Avaliando Modelo
scores = model.evaluate(X, Y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


# Resumo da Rede
print(model.summary())
