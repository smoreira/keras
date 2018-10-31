import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split

np.random.seed(7)

#carregando o dataset
dataset = pd.read_csv("Churn_Modelling.csv", header=0)

# Teste: 
#print (dataset.head())

# transformar valores de sexo e local em inteiros
if dataset["Gender"][0] in ("Male", "Female"):
    for row in [dataset]:
        row["Gender"] = row["Gender"].map( {"Male": 0, "Female": 1} ).astype(int)
        row["Geography"] = row["Geography"].map( {"France": 0, "Germany": 1, "Spain": 2} ).astype(int)

# transformar faixa de idade 
if dataset["Age"][0] > 17:
    max_age = dataset["Age"].max()
    # mapping
    dataset["Age"] = dataset["Age"] * (4 / max_age)

# transformar CreditScore
if dataset["CreditScore"][0] > 4:
    max_creditScore = dataset["CreditScore"].max()
    # mapping
    dataset["CreditScore"] = dataset["CreditScore"] * (4 / max_creditScore)

# transformar Balance
if dataset["Balance"][1] > 4:
    max_balance = dataset["Balance"].max()
    # mapping
    dataset["Balance"] = dataset["Balance"] * (4 / max_balance)

# transformar Salario Estimado
if dataset["EstimatedSalary"][0] > 4:
    max_estimatedSalary = dataset["EstimatedSalary"].max()
    # mapping
    dataset["EstimatedSalary"] = dataset["EstimatedSalary"] * (4 / max_estimatedSalary)

# transformar Tenure
if dataset["Tenure"][0] > 0:
    max_tenure = dataset["Tenure"].max()
    # mapping
    dataset["Tenure"] = dataset["Tenure"] * (4 / max_tenure)

#deletando as 3 primeiras colunas
if "RowNumber" in dataset.columns:
    dataset = dataset.drop(["RowNumber"], axis=1)
if "CustomerId" in dataset.columns:
    dataset = dataset.drop(["CustomerId"], axis=1)
if "Surname" in dataset.columns:
    dataset = dataset.drop(["Surname"], axis=1)

#separando valores de X e Y
X = dataset.iloc[:,0:10]
Y = dataset.iloc[:,10]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33)


#Criando Modelo (fully-connected layers)
model = Sequential()
model.add(Dense(64, kernel_initializer='uniform', activation='relu', input_dim=10))
model.add(Dense(32, kernel_initializer='uniform', activation='relu'))
model.add(Dense(64, kernel_initializer='uniform', activation='relu'))
model.add(Dense(32, kernel_initializer='uniform', activation='relu'))
model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

#Compilando
model.compile(loss='binary_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

#Treinando
model.fit(X_train, Y_train,
          epochs=100,
          batch_size=20)

#Avaliando
scores = model.evaluate(X_test, Y_test, batch_size=10, verbose=2)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

Xnew = np.array([[608,2,1,41,1,83807.86,1,0,1,112542.58]]) #0
Xnew1 = np.array([[622,2,1,46,4,107073.27,2,1,1,30984.59]]) #1

prediction = model.predict_classes(Xnew) #array
prediction1 = model.predict_classes(Xnew1) #array

print("Predicted/0 = %s, Predicted/1 = %s" % (prediction[0], prediction1[0]))

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")


