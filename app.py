import streamlit as st
import pandas as pd
import joblib

st.title("Predicción de especie de flor Iris")

st.write("Nombre: TU NOMBRE")
st.write("Código ISIL: TU CÓDIGO")
st.write("[Ver cuaderno en Google Colab](PEGA_AQUI_TU_LINK_DE_COLAB)")

modelo = joblib.load("modelos/modelo_random_forest.pkl")

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
