import numpy as np
import streamlit as st
import pandas as pd

st.write(''' # Predicción de Costos ''')
st.image("euler.jpg", caption="Analicemos cuánto gastarás día con día.")

st.header('Datos del gasto')

def user_input_features():
    # Entrada
    Presupuesto = st.number_input('Presupuesto:', min_value=0.0, max_value=500.0, value=0.0, step=1.0)
    Tiempo_invertido = st.number_input('Tiempo invertido:', min_value=0, max_value=100, value=0, step=1)
    Tipo = st.number_input('Tipo:', min_value=1, max_value=10, value=1, step=1)
    Momento = st.number_input('Momento:', min_value=1, max_value=10, value=1, step=1)
    No_personas = st.number_input('No. de personas:', min_value=1, max_value=20, value=1, step=1)

    user_input_data = {'Presupuesto': Presupuesto,
                       'Tiempo invertido': Tiempo_invertido,
                       'Tipo': Tipo,
                       'Momento': Momento,
                       'No. de personas': No_personas,
                       }

    features = pd.DataFrame(user_input_data, index=[0])
    return features

df = user_input_features()

datos = pd.read_csv('Gastos.csv', encoding='utf-8')
X = datos.drop(columns='Costo')
y = datos['Costo']

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1614954)
LR = LinearRegression()
LR.fit(X_train, y_train)

b1 = LR.coef_
b0 = LR.intercept_
prediccion = b0 + b1[0]*df['Presupuesto'] + b1[1]*df['Tiempo invertido'] + b1[2]*df['Tipo'] + b1[3]*df['Momento'] + b1[4]*df['No. de personas']

st.subheader('Cálculo del Costo')
st.write('El costo estimado es: $', round(float(prediccion), 2))

# Métricas del modelo
from sklearn.metrics import r2_score, mean_squared_error
y_pred = LR.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

st.subheader('Métricas del Modelo')
st.write(f'R² Score: {r2:.4f}')
st.write(f'RMSE: ${rmse:.2f}')
