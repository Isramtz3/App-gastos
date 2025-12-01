import numpy as np
import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


st.write('''# Predicción del costo de actividad''')
st.image("euler,jpg", caption="Analicemos si gastarás mucho en tu día a día.")

st.header('Datos de evaluación')

def user_input_features():
    Presupuesto = st.number_input('Presupuesto:', min_value=1.0, max_value=100000.0, value=20.0, step=0.1)
    Tiempo_invertido = st.number_input('Tiempo invertido:', min_value=1.0, max_value=1400.0, value=1.0, step=1.0)
    Tipo = st.number_input('Tipo (1-6):', min_value=1, max_value=6, value=1, step=1)
    Momento = st.number_input('Momento (1-3):', min_value=1, max_value=3, value=1, step=1)
    No_personas = st.number_input('No. de personas:', min_value=1, max_value=100, value=1, step=1)

    user_input_data = {'Presupuesto': Presupuesto,
                       'Tiempo invertido': Tiempo_invertido,
                       'Tipo': Tipo,
                       'Momento': Momento,
                       'No. de personas': No_personas}

    features = pd.DataFrame(user_input_data, index=[0])
    return features

df = user_input_features()

gastos = pd.read_csv('Gastos_ok (1).csv', encoding='utf-8')
X = gastos.drop(columns='Costo')
Y = gastos['Costo']

classifier = DecisionTreeClassifier(max_depth=2, criterion='squared_error', min_samples_leaf=25, max_features=1, random_state=1615170)
classifier.fit(X, Y)

prediction = classifier.predict(df)

st.subheader('Predicción')
st.write(f'Costo estimado: ${prediction[0]:.2f}')
