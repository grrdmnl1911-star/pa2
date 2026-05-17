import streamlit as st
import pandas as pd
import joblib

# ==============================
# TÍTULO
# ==============================

st.title("Predicción de especie de flor Iris")

# ==============================
# DATOS DEL ESTUDIANTE
# ==============================

st.write("Nombre: Kenia Alexis Sandoval Toledo")
st.write("Código ISIL: TU_CODIGO")

st.write(
    "[Ver cuaderno en Google Colab](https://colab.research.google.com/drive/1pTMeKeg5Lo6yUd3Us0BLZKuKlQ3ncBdw?usp=sharing)"
)

# ==============================
# CARGAR MODELO
# ==============================

modelo = joblib.load("modelos/modelo_random_forest.pkl")

# ==============================
# INPUTS
# ==============================

st.subheader("Ingrese los datos de la flor")

sepal_length = st.number_input(
    "Largo del sépalo (cm)",
    0.0,
    10.0,
    5.1
)

sepal_width = st.number_input(
    "Ancho del sépalo (cm)",
    0.0,
    10.0,
    3.5
)

petal_length = st.number_input(
    "Largo del pétalo (cm)",
    0.0,
    10.0,
    1.4
)

petal_width = st.number_input(
    "Ancho del pétalo (cm)",
    0.0,
    10.0,
    0.2
)

# ==============================
# DATAFRAME
# ==============================

datos = pd.DataFrame(
    [[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]],
    columns=[
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)"
    ]
)

# ==============================
# PREDICCIÓN
# ==============================

if st.button("Predecir especie"):

    prediccion = modelo.predict(datos)[0]

    especies = [
        "setosa",
        "versicolor",
        "virginica"
    ]

    st.success(
        f"La especie predicha es: {especies[prediccion]}"
    )
