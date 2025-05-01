import streamlit as st
import joblib
import pandas as pd

# Load the saved model
model = joblib.load('iris_model.pkl')
label_encoder = joblib.load('label_encoder.pkl') 
# Streamlit app layout
st.title('Iris Species Prediction')

# User input
sepal_length = st.number_input('Sepal Length', min_value=0.0, max_value=10.0, value=5.0)
sepal_width = st.number_input('Sepal Width', min_value=0.0, max_value=10.0, value=3.0)
petal_length = st.number_input('Petal Length', min_value=0.0, max_value=10.0, value=4.0)
petal_width = st.number_input('Petal Width', min_value=0.0, max_value=10.0, value=1.0)
#sepal_length = st.text_input('Enter Sepal Length')
#sepal_width = st.text_input('Enter Sepal Width')
#petal_length = st.text_input('Enter Petal Length')
#petal_width = st.text_input('Enter Petal Width')

# Prediction button
if st.button('Predict'):
    try:
        # Convert user input to float
        features = [[float(sepal_length), float(sepal_width), float(petal_length), float(petal_width)]]
        
        # Make prediction
        prediction = model.predict(features)
        predicted_label = label_encoder.inverse_transform(prediction)

    # Display the result
        st.write(f"The predicted Iris species is: {predicted_label[0]}")
        #st.write(f'The predicted species is: {prediction[0]}')
    except Exception as e:
        st.error(f'Error in prediction: {e}')