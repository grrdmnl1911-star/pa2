import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Título
st.title("Predicción de especie de flor Iris")

# Datos del estudiante
st.write("Nombre: Kenia Alexis Sandoval Toledo")
st.write("Código ISIL: TU_CODIGO_ISIL")

# Link de Google Colab
st.write(
    "[Ver cuaderno en Google Colab](https://colab.research.google.com/drive/1pTMeKeg5Lo6yUd3Us0BLZKuKlQ3ncBdw?usp=sharing)"
)

# Cargar dataset Iris
iris = load_iris()

X = iris.data
y = iris.target

# Entrenar modelo
modelo = RandomForestClassifier(random_state=42)
modelo.fit(X, y)

# Inputs del usuario
st.subheader("Ingrese los datos de la flor")

sepal_length = st.number_input(
    "Largo del sépalo (cm)",
    min_value=0.0,
    max_value=10.0,
    value=5.1
)

sepal_width = st.number_input(
    "Ancho del sépalo (cm)",
    min_value=0.0,
    max_value=10.0,
    value=3.5
)

petal_length = st.number_input(
    "Largo del pétalo (cm)",
    min_value=0.0,
    max_value=10.0,
    value=1.4
)

petal_width = st.number_input(
    "Ancho del pétalo (cm)",
    min_value=0.0,
    max_value=10.0,
    value=0.2
)

# Crear DataFrame
datos = pd.DataFrame(
    [[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]],
    columns=iris.feature_names
)

# Predicción
if st.button("Predecir especie"):

    prediccion = modelo.predict(datos)[0]
    especie = iris.target_names[prediccion]

    st.success(f"La especie predicha es: {especie}")

    if especie == "setosa":
        st.balloons()
