import streamlit as st
import pandas as pd
import joblib
import pickle


st.title("Predicción de especie de flor Iris")

st.write("Nombre: Gerardo Arias Alzamora")
st.write("Código ISIL: 45443379")
st.write("[Ver cuaderno en Google Colab](https://colab.research.google.com/drive/1pTMeKeg5Lo6yUd3Us0BLZKuKlQ3ncBdw?usp=sharing)")

with open("modelos/modelo_random_forest.pkl", "rb") as archivo:
    modelo = pickle.load(archivo)

st.subheader("Ingrese los datos de la flor")

sepal_length = st.number_input("Largo del sépalo (cm)", 0.0, 10.0, 5.1)
sepal_width = st.number_input("Ancho del sépalo (cm)", 0.0, 10.0, 3.5)
petal_length = st.number_input("Largo del pétalo (cm)", 0.0, 10.0, 1.4)
petal_width = st.number_input("Ancho del pétalo (cm)", 0.0, 10.0, 0.2)

datos = pd.DataFrame({
    "sepal length (cm)": [sepal_length],
    "sepal width (cm)": [sepal_width],
    "petal length (cm)": [petal_length],
    "petal width (cm)": [petal_width]
})

if st.button("Predecir"):
    prediccion = modelo.predict(datos)[0]

    especies = {
        0: "Setosa",
        1: "Versicolor",
        2: "Virginica"
    }

    st.success(f"La especie predicha es: {especies[prediccion]}")
