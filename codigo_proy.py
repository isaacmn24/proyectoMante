import winsound
import tensorflow as tf
import pandas as pd
import numpy as np
from keras import layers, models, optimizers, utils, Input, regularizers
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from scikeras.wrappers import KerasRegressor
import os  # Para manejar las carpetas
import time

# Hiperparámetros
train_percentage = 0.5
n = 1
units = 30
activation = "sigmoid"
learning_rate = 0.003
loss = 'mse'
batch_size = 200
epochs = 80
l2_lambda = 0
dropout_rate = 0

# Lectura de los datos
train_dataset = pd.read_excel("data/train.xlsx")
test_dataset = pd.read_excel("data/test.xlsx")

# Se asignan los datos de entrenamiento y validación

X_train = train_dataset.copy()
X_train = X_train.drop(columns='TIME')

Y_train = train_dataset.copy()
Y_train = Y_train['TIME']

X_test = test_dataset.copy()
X_test = X_test.drop(columns='TIME')

Y_test = test_dataset.copy()
Y_test = Y_test['TIME']

print("Forma de X_train:", X_train.shape)
print("Forma de X_test:", X_test.shape)

print(X_test)

def modelo(X_train, X_test, Y_train, Y_test, dropout):
    # Definir el modelo
    model = models.Sequential()

    model.add(Input(shape=(25,)))

    for _ in range(n):
        model.add(layers.Dense(units=units, activation=activation,
                               kernel_regularizer=regularizers.l2(l2_lambda)))

    # Capa de salida con una salida y activación lineal para regresión
    model.add(layers.Dense(
                units=1,  # Número de neuronas de salida
                activation='linear',  # Activación lineal para problemas de regresión
                kernel_regularizer=regularizers.l2(l2_lambda)))

    # Compilar el modelo
    model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate),
                  loss=loss,  # Establecer la función de pérdida
                  metrics=['mae'])  # Métrica de precisión

    inicio_tiempo = time.time()

    losses = model.fit(x=X_train,  # Datos de entrada (entrenamiento)
                     y=Y_train,  # Datos de salida (entrenamiento)
                     validation_data=( # Conjuntos de datos de validación
                            X_test, # Entrada de validación
                            Y_test  # Salida de validación
                            ),
                     batch_size=batch_size,  # Tamaño de muestreo
                     epochs=epochs  # Cantidad de iteraciones de entrenamiento
                     )
    tiempo_entrenamiento = time.time() - inicio_tiempo

    # Mostrar el resumen del modelo
    model.summary()
    return model, losses, tiempo_entrenamiento

# Crear carpeta para almacenar los resultados con los hiperparámetros en el nombre
nombre_carpeta = f"resultados_pruebas_param"
if not os.path.exists(nombre_carpeta):
    os.makedirs(nombre_carpeta)

# Caso B: Error / Colisión / Obstrucción / Movimiento-Pérdida
model, losses, tiempo_entrenamiento= modelo(X_train, X_test, Y_train, Y_test, dropout_rate)

loss_df = pd.DataFrame(losses.history)
loss_df.loc[:, ['loss', 'val_loss']].plot()
plt.title(f"Curvas de entrenamiento B\nactve: {activation}, layers: {n}, train: {train_percentage}, units: {units}, lr: {learning_rate}, l2: {l2_lambda}")

# Guardar el modelo entrenado y los pesos
plt.savefig(f"{nombre_carpeta}/curvas_entrenamiento_{activation}_lay_{n}_units_{units}_tp_{train_percentage}lr_{learning_rate}_l2_{l2_lambda}.png")
model.save(f"{nombre_carpeta}/modelo_entrenado_2.h5")

print('tiempo de entrenamiento: ', tiempo_entrenamiento)
print('loss: ', loss_df['loss'][epochs-1])
print('va_loss: ', loss_df['val_loss'][epochs-1])
print(f"Modelo guardado en {nombre_carpeta}/modelo_entrenado_2.h5")

val_x = X_test
predicciones = model.predict(val_x)
val_y = np.array(Y_test)
predicciones = np.array(predicciones)

print('Valores reales: ')
print(val_y)
print('\nValores predecidos: ')
print(predicciones)

plt.show()


