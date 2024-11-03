# Importar librerías necesarias
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from scikeras.wrappers import KerasRegressor
import tensorflow as tf
from keras import layers, models, optimizers, Input, regularizers

# Lectura de los datos
train_dataset = pd.read_excel("data/train.xlsx")
test_dataset = pd.read_excel("data/test.xlsx")

# Preparar los datos de entrenamiento y prueba
X_train = train_dataset.drop(columns='TIME')
Y_train = train_dataset['TIME'].values.ravel()  # Convertir a vector unidimensional
X_test = test_dataset.drop(columns='TIME')
Y_test = test_dataset['TIME'].values.ravel()  # Convertir a vector unidimensional

# Función para crear el modelo
def create_model(neurons=64, learning_rate=0.001, layers=2):
    model = models.Sequential()
    model.add(layers.Dense(neurons, input_dim=X_train.shape[1], activation='relu'))
    
    for _ in range(layers - 1):
        model.add(layers.Dense(neurons, activation='relu'))
    
    model.add(layers.Dense(1))  # Capa de salida
    model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate), loss='mse', metrics=['mae'])
    return model

# Definir el modelo con KerasRegressor
model = KerasRegressor(model=create_model, verbose=0)

# Definición del espacio de búsqueda de hiperparámetros
param_dist = {
    'neurons': [32, 64, 128],
    'learning_rate': [0.01, 0.001, 0.0001],
    'layers': [1, 2, 3],
    'batch_size': [16, 32, 64],
    'epochs': [50, 100, 200]
}

# Búsqueda de hiperparámetros con RandomizedSearchCV
random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=10, cv=3, scoring='neg_mean_absolute_error', random_state=42)
random_search_result = random_search.fit(X_train, Y_train)

# Mostrar resultados de la búsqueda
print("Mejores hiperparámetros: ", random_search_result.best_params_)
print("Mejor desempeño (MAE): ", -random_search_result.best_score_)

# Entrenar el mejor modelo
best_model = random_search_result.best_estimator_
history = best_model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=random_search_result.best_params_['epochs'], batch_size=random_search_result.best_params_['batch_size'], verbose=1)

# Predicciones
predicciones = best_model.predict(X_test)

# Resultados
print('Valores reales: ')
print(Y_test)
print('\nValores predecidos: ')
print(predicciones)
