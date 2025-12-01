import numpy as np
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

st.write(''' # Predicción de Costos ''')
st.image("https://placehold.co/600x150/0000FF/FFFFFF?text=Analizando+Gastos", caption="Analicemos cuánto dinero gastarás en tu día a día.")
st.header('Datos')

def user_input_features():
    Presupuesto = st.number_input('Presupuesto:', min_value=0.0, max_value=5000.0, value=0.0, step=1.0)
    Tiempo_invertido = st.number_input('Tiempo invertido:', min_value=0, max_value=1000, value=10, step=1)
    Tipo = st.number_input('Tipo (1-6):', min_value=1, max_value=6, value=6, step=1)
    Momento = st.number_input('Momento (1-3):', min_value=1, max_value=3, value=3, step=1)
    No_personas = st.number_input('No. de personas:', min_value=1, max_value=20, value=5, step=1)
    
    user_input_data = {'Presupuesto': Presupuesto,
                       'Tiempo invertido': Tiempo_invertido,
                       'Tipo': Tipo,
                       'Momento': Momento,
                       'No. de personas': No_personas
                       }
    features = pd.DataFrame(user_input_data, index=[0])
    return features

df = user_input_features()

try:
    f = pd.read_csv("Gastos.csv")
except FileNotFoundError:
    st.error("Error: No se encontró el archivo 'Gastos.csv'. Asegúrate de que está en el directorio correcto.")
    f = pd.DataFrame({
        'Presupuesto': [100, 200, 50, 300, 150, 0, 500],
        'Tiempo invertido': [5, 12, 3, 20, 8, 15, 25],
        'Tipo': [1, 3, 6, 2, 4, 5, 1],
        'Momento': [1, 2, 3, 1, 3, 2, 1],
        'No. de personas': [2, 4, 1, 5, 3, 2, 6],
        'Costo': [25, 60, 10, 80, 45, 12, 150]
    })


X = f[['Presupuesto', 'Tiempo invertido', 'Tipo', 'Momento', 'No. de personas']]
y = f['Costo']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1615170)

LR = LinearRegression()
LR.fit(X_train, y_train)

b1 = LR.coef_
b0 = LR.intercept_

prediccion_raw = b0 + b1[0]*df['Presupuesto'] + b1[1]*df['Tiempo invertido'] + b1[2]*df['Tipo'] + b1[3]*df['Momento'] + b1[4]*df['No. de personas']

prediccion_ajustada = np.maximum(0, prediccion_raw)

st.subheader('Predicción del Costo')
st.write('El costo estimado es: $', round(float(prediccion_ajustada), 2))


y_pred = LR.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

st.subheader('Métricas del Modelo')
st.write(f'R² Score: {r2:.4f}')
st.write(f'RMSE: {rmse:.2f}')
