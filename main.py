import streamlit as st
import pandas as pd
import datetime
import xgboost as xgb
import pathlib

# Function to load CSS from the 'assets' folder
def load_css(file_path):
    with open(file_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load the external CSS
css_path = pathlib.Path("assets/styles.css")
load_css(css_path)

# Load pre-trained model
date_time = datetime.datetime.now()
model = xgb.XGBRegressor()
model.load_model('xgb_model.json')

# Main function
def main():
    # Header Section
    st.markdown(
        """
        <div class='header-container'>
            <h1>Car Price Prediction</h1>
            <p>Get an accurate estimate for your car's price</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Enter Your Car Details Section
    st.markdown("<h3 id='details-section' class='details-header'>Enter Your Car Details:</h3>", unsafe_allow_html=True)
    
    # Inputs (form fields without a container)
    p1 = st.number_input("Ex-showroom Price (in Lakhs):", min_value=2.5, max_value=25.0, step=0.1)
    p2 = st.number_input("Distance Driven (in KM):", min_value=100, max_value=500000, step=100)
    
    s1 = st.selectbox("Fuel Type:", ["Petrol", "Diesel", "CNG"])
    p3 = {"Petrol": 0, "Diesel": 1, "CNG": 2}[s1]
    
    s2 = st.selectbox("Seller Type:", ["Dealer", "Individual"])
    p4 = {"Dealer": 0, "Individual": 1}[s2]
    
    s3 = st.selectbox("Transmission Type:", ["Manual", "Automatic"])
    p5 = {"Manual": 0, "Automatic": 1}[s3]
    
    p6 = st.slider("Number of Owners:", min_value=0, max_value=3)
    years = st.number_input("Year of Purchase:", min_value=1990, max_value=date_time.year, step=1)
    p7 = date_time.year - years

    data_new = pd.DataFrame({
        "Present_Price": p1,
        "Kms_Driven": p2,
        "Fuel_Type": p3,
        "Seller_Type": p4,
        "Transmission": p5,
        "Owner": p6,
        "Age": p7,
    }, index=[0])

    # Predict Button
    if st.button("Predict Price"):
        try:
            prediction = model.predict(data_new)
            if prediction > 0:
                st.balloons()
                st.success(f"You can sell the car for approximately ₹{prediction[0]:.2f} Lakhs.")
            else:
                st.warning("You might not be able to sell this car at a profitable price.")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

    # Footer Section
    st.markdown(
        """
        <div class='footer-container'>
            <p>Made with ❤️ using Streamlit</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    main()
