import numpy as np
import streamlit as st
import pandas as pd

st.write(''' # Predicción de Costo de actividad  ''')
st.image("euler.jpg", caption="Analicemos cuánto gastarás día con día.")
st.header('Datos')

def user_input_features():
    # Entrada
    Presupuesto = st.number_input('Presupuesto:', min_value=0.0, max_value=5000000.0, value=0.0, step=1.0)
    Tiempo_invertido = st.number_input('Tiempo invertido (minutos):', min_value=0, max_value=1000, value=0, step=1)
    Tipo = st.number_input('Tipo (1-6):', min_value=1, max_value=6, value=1, step=1)
    Momento = st.number_input('Momento (1-3):', min_value=1, max_value=3, value=1, step=1)
    No_personas = st.number_input('No. de personas:', min_value=1, max_value=100, value=1, step=1)
    
    user_input_data = {'Presupuesto': Presupuesto,
                       'Tiempo invertido': Tiempo_invertido,
                       'Tipo': Tipo,
                       'Momento': Momento,
                       'No. de personas': No_personas
                       }
    features = pd.DataFrame(user_input_data, index=[0])
    return features

df = user_input_features()

# Leer el archivo CSV
f = pd.read_csv("Gastos.csv")

# Preparar los datos
X = f[['Presupuesto', 'Tiempo invertido', 'Tipo', 'Momento', 'No. de personas']]
y = f['Costo']

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Dividir los datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1615170)

# Entrenar el modelo
LR = LinearRegression()
LR.fit(X_train, y_train)

# Obtener coeficientes
b1 = LR.coef_
b0 = LR.intercept_

# Hacer predicción
prediccion = b0 + b1[0]*df['Presupuesto'] + b1[1]*df['Tiempo invertido'] + b1[2]*df['Tipo'] + b1[3]*df['Momento'] + b1[4]*df['No. de personas']

st.subheader('Predicción del Costo')
if prediccion < 0:
    st.write("El costo fue negativo (no tiene sentido), así que se reduce a $0"
             else:
    st.write('El costo estimado es: $', round(float(prediccion), 2))

# Mostrar métricas del modelo
from sklearn.metrics import r2_score, mean_squared_error
y_pred = LR.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

st.subheader('Métricas del Modelo')
st.write(f'R² :) : {r2:.4f}')
st.write(f'RMSE: {rmse:.2f}')
