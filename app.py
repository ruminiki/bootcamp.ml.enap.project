import pandas as pd
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
sepal_length = st.slider("Sepal Length (cm)", 0.0, 10.0, 5.0)  # Adjust min and max as needed
sepal_width = st.slider("Sepal Width (cm)", 0.0, 10.0, 3.0)    # Adjust min and max as needed
petal_length = st.slider("Petal Length (cm)", 0.0, 10.0, 1.5)  # Adjust min and max as needed
petal_width = st.slider("Petal Width (cm)", 0.0, 10.0, 0.2)    # Adjust min and max as needed

# Button for prediction
if st.button("Predict"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    
    # Get predicted probabilities for each class
    probabilities = model.predict_proba(input_data)[0]
    
    # Assuming you have a mapping of prediction to species names
    target_names = ['setosa', 'versicolor', 'virginica']
    
    # Create a DataFrame for better visualization
    prob_df = pd.DataFrame({
        'Species': target_names,
        'Probability': probabilities
    })
    
    # Display the probabilities in a bar chart
    st.bar_chart(prob_df.set_index('Species'))

    # Show the predicted species based on highest probability
    predicted_species = target_names[np.argmax(probabilities)]
    st.write(f"Predicted species: {predicted_species}")
