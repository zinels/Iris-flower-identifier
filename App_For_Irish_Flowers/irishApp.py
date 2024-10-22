import streamlit as st
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Load the trained k-NN model using joblib
knn_model = joblib.load('knn_model.pkl')

# Load the saved StandardScaler (fitted on the training data)
scaler = joblib.load('scaler.pkl')

# Mapping label to species name
species_mapping = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}

# Define a function to predict the flower species
def predict_species(sepal_length, sepal_width, petal_length, petal_width):
    # Create a 2D array with the input values
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    # Scale the features using the loaded, fitted scaler
    scaled_features = scaler.transform(features)

    # Predict the species
    prediction = knn_model.predict(scaled_features)
    species = species_mapping[prediction[0]]
    return species

#st.image('bg.png', caption='Iris Flower', use_column_width=True)

# Streamlit app layout
st.title("Iris Flower Species Prediction")
st.write("Enter the characteristics of the Iris flower below:")

# Input fields for flower measurements
sepal_length = st.slider("Sepal Length (cm)", min_value=4.0, max_value=8.0, value=5.1, step=0.1)
sepal_width = st.slider("Sepal Width (cm)", min_value=2.0, max_value=4.5, value=3.5, step=0.1)
petal_length = st.slider("Petal Length (cm)", min_value=1.0, max_value=7.0, value=1.4, step=0.1)
petal_width = st.slider("Petal Width (cm)", min_value=0.1, max_value=2.5, value=0.2, step=0.1)

# When the user clicks the "Predict" button
if st.button("Predict"):
    species = predict_species(sepal_length, sepal_width, petal_length, petal_width)
    st.write(f"The predicted species is: **{species}**")
