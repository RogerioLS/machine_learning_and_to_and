import streamlit as st
import pickle
import pandas as pd
from components.sidebar import render_sidebar

# Load the trained model
model = pickle.load(open('model_trained/model_regression_immobile.pkl', 'rb'))

def main():
    render_sidebar()

    st.title("Real Estate Price Prediction in California")
    st.write("Input the features below to predict the price:")

    # Input features
    MedInc = st.number_input("Median Income", value=3.0)
    HouseAge = st.number_input("House Age", value=20)
    AveRooms = st.number_input("Average Rooms", value=5)
    AveBedrms = st.number_input("Average Bedrooms", value=1)
    Population = st.number_input("Population", value=300)
    AveOccup = st.number_input("Average Occupancy", value=3)
    Latitude = st.number_input("Latitude", value=34.0)
    Longitude = st.number_input("Longitude", value=-118.0)

    # Create a dictionary to hold the inputs
    input_data = {
        'MedInc': MedInc,
        'HouseAge': HouseAge,
        'AveRooms': AveRooms,
        'AveBedrms': AveBedrms,
        'Population': Population,
        'AveOccup': AveOccup,
        'Latitude': Latitude,
        'Longitude': Longitude
    }

    # Predict button
    if st.button("Predict"):
        # Convert dictionary to DataFrame
        input_df = pd.DataFrame([input_data])

        # Check the shape of the input dataframe
        st.write("Input DataFrame:")
        st.write(input_df)

        # Predict using the loaded model
        try:
            prediction = model.predict(input_df)
            st.write(f"Predicted Price: ${prediction[0]:,.2f}")
        except ValueError as e:
            st.error(f"Error in prediction: {e}")

if __name__ == "__main__":
    main()