import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# ==============================
# TITULO
# ==============================

st.title("Predicción de especie de flor Iris")

st.write("Nombre: Kenia Alexis Sandoval Toledo")
st.write("Código ISIL: TU_CODIGO")

st.write(
    "[Ver cuaderno en Google Colab](https://colab.research.google.com/drive/1pTMeKeg5Lo6yUd3Us0BLZKuKlQ3ncBdw?usp=sharing)"
)

# ==============================
# DATASET
# ==============================

iris = load_iris()

X = iris.data
y = iris.target

# ==============================
# MODELO RANDOM FOREST
# ==============================

modelo_rf = RandomForestClassifier(random_state=42)

modelo_rf.fit(X, y)

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
# PREDICCION
# ==============================

if st.button("Predecir especie"):

    prediccion = modelo_rf.predict(datos)[0]

    especies = [
        "setosa",
        "versicolor",
        "virginica"
    ]

    st.success(
        f"La especie predicha es: {especies[prediccion]}"
    )
