import streamlit as st
import numpy as np
import joblib

@st.cache_resource
def load_model():
    
    return joblib.load('best_model.pkl') 

model = load_model()
class_names = ['Setosa', 'Versicolor', 'Virginica']

st.title(" Iris Flower Species Predictor")
st.write("Enter feature values below to classify the Iris flower species.")

sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.8)
sepal_width  = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.0)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 4.35)
petal_width  = st.slider("Petal Width (cm)", 0.1, 2.5, 1.3)

if st.button("Predict Species"):
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(features)[0]
    predicted_class = class_names[prediction]
    
    st.success(f"**Predicted Species:** {predicted_class}")
