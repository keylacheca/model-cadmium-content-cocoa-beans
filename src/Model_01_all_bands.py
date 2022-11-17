import pandas as pd
import numpy as np
import tensorflow as tf
import os.path
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from keras.layers import LeakyReLU
from keras import regularizers
import seaborn as sns
from sklearn.metrics import r2_score

#tf.compat.v1.disable_eager_execution()

#filePath = "DataFinalHP8Mix2.csv"
filePath = "DataFinalHPF5Mix_hasta3_Good.csv"
#df_0 = pd.read_csv(filePath)
#df_shuffled:
#df = df_0.sample(frac=1).reset_index(drop=True)

df = pd.read_csv(filePath)

print(df.describe())

X = df.drop(['cadmio','muestra'], axis=1)

Y = df['cadmio']
columns = ['cadmio']
y = pd.DataFrame(Y, columns=columns)

#CROSS VALIDATION: TRAINING SET AND TEST SET
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state=2)
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.fit_transform(X_train)  # .fit_trasnform se usa en data train para conocer la varianza y media en estos datos
X_test = scaler.transform(X_test)        # .transform se usa en el data test para aplicar la misma varianza y media de data train

#MODELING:LAYERS, NEURONS ANDA ACTVIVATION FUNCTIONS
model = Sequential()
model.add(Dense(units=50, input_dim = 240))
model.add(Dropout(0.6))
model.add(Dense(units =25,activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(units =10, activation="relu"))
model.add(Dense(units =5, activation="relu"))
# NORMALIZER: DROPOUT AND EARLYSTOPPING
#Output layer
model.add(Dense(units =1, activation="linear"))

#OPTIMIZER: ADAMS
model.compile(optimizer = "adam", loss="mean_squared_error", metrics=["mae","mse"])
print(model.summary())

# EarlyStopping
callbacks = [EarlyStopping(monitor="loss", patience=100, verbose = 0)]

modelo = model.fit(X_train, y_train, batch_size = 32, epochs = 2000, 
                 validation_data=(X_test, y_test))

loss = modelo.history["loss"]
val_loss = modelo.history["val_loss"]
epochs = range(1, len(loss)+1)
p1=plt.plot(epochs, loss, "--b", label = "Loss entrenamiento")
plt.plot(epochs, val_loss, "r", label = "Loss validación")
plt.title("Loss de entrenamiento y validación")
plt.xlabel("Épocas")
plt.ylabel("Loss")
plt.legend()
plt.show(p1)

p2= plt.plot(epochs, np.log10(loss), "y", label = "Log Loss entrenamiento")
plt.plot(epochs, np.log10(val_loss), "r", label = "Log Loss validación")
plt.title("Log loss")
plt.xlabel("Épocas")
plt.ylabel("Loss")
plt.legend()
plt.show(p2)

acc = modelo.history["mae"]
val_acc = modelo.history["val_mae"]
plt.plot(epochs, acc, "--b", label = "MAE entrenamiento")
plt.plot(epochs, val_acc, "r", label = "MAE validación")
plt.title("MAE de entrenamiento y validación")
plt.xlabel("Épocas")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

acc = modelo.history["mse"]
val_acc = modelo.history["val_mse"]
plt.plot(epochs, acc, "--b", label = "MSE entrenamiento")
plt.plot(epochs, val_acc, "r", label = "MSE validación")
plt.title("MSE de entrenamiento y validación")
plt.xlabel("Épocas")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

y_predTrain = model.predict(X_train)[:,0]
y_predTest = model.predict(X_test)[:,0]

plt.scatter(y_test,y_predTest, marker='.',linestyle="None", color='b')
plt.scatter(y_train,y_predTrain, marker='.',linestyle="None", color='r')
#plt.xlabel("Real value")
#plt.ylabel("Predicted value")
plt.xlabel("Valor real")
plt.ylabel("Valor predicho")
plt.xlim(0,2)
plt.ylim(0,2)
plt.show()

p4 = sns.residplot(x= y_predTest, y=y_test, data=df, lowess=True,
                                  line_kws={'color': 'red', 'lw': 1, 'alpha': 1})

#plt.xlabel("Fitted values")
#plt.title('Residual plot')
plt.xlabel("Valores ajustados")
plt.title('Gráfico de residuos')

plt.show()

print(r2_score(y_test, y_predTest))
print(r2_score(y_train, y_predTrain))

model.save('Model_cocoa_beans_N_02_01.h5')