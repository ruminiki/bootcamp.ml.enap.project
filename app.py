import streamlit as st
import pickle
import numpy as np

# Carregar o modelo salvo em .pkl
with open("random_forest_model.pkl", "rb") as f:
    model = pickle.load(f)

# Título da aplicação
st.title("Classificação de Flores Iris")

# Descrição
st.write("Modelo: Classificador de flores Iris (Setosa, Versicolor, Virginica).")
st.write("Dados de treinamento: Conjunto de dados Iris.")
st.write("Autor: Ruminiki Schmoeller.")
st.write(" ")
st.write("Insira os valores para fazer uma classificação:")

# Input sliders for sepal length, sepal width, petal length, and petal width
sepal_length = st.sidebar.slider("Sepal Length (cm)", 0.0, 10.0, 5.0)  # Adjust min and max as needed
sepal_width = st.sidebar.slider("Sepal Width (cm)", 0.0, 10.0, 3.0)    # Adjust min and max as needed
petal_length = st.sidebar.slider("Petal Length (cm)", 0.0, 10.0, 1.5)  # Adjust min and max as needed
petal_width = st.sidebar.slider("Petal Width (cm)", 0.0, 10.0, 0.2)    # Adjust min and max as needed

# Button for prediction
if st.button("Predict"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)
    
    # Assuming you have a mapping of prediction to species names
    target_names = ['setosa', 'versicolor', 'virginica']
    prediction_species = target_names[prediction[0]]
    
    prediction_proba = model.predict_proba(input_data)

    # Dicionário das espécies
    species = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
    
    # Exibir o resultado
    st.markdown(f"<h4><b>A flor é da espécie: {species[prediction[0]]}</b></h4>", unsafe_allow_html=True)
    
    # Exibir as probabilidades formatadas
    st.write("Probabilidades:")
    for i, prob in enumerate(prediction_proba[0]):
        st.write(f"{species[i]}: {prob:.2%}")
