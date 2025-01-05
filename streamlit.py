import streamlit as st
import pandas as pd
import numpy as np

# Load data
@st.cache_data  # Disarankan menggunakan cache_data untuk penyimpanan data
def load_data():
    data_location = './mmm.csv'  # Pastikan path ini benar
    data = pd.read_csv(data_location)
    return data

row_data = load_data()

# Sidebar navigation
st.sidebar.title("Navigation")
pages = ["Home", "Data Overview", "Model Training", "Prediction"]
selection = st.sidebar.radio("Go to", pages)

# Home Page
if selection == "Home":
    st.title("Used Car Price Prediction")
    st.write("""
    Welcome to the Used Car Price Prediction App! 
    Use the navigation on the left to explore the app features.
    """)

# Data Overview Page
elif selection == "Data Overview":
    st.title("Data Overview")
    st.write("### Raw Data")
    st.dataframe(row_data.head())
    st.write("### Data Shape")
    st.write(row_data.shape)
    st.write("### Data Description")
    st.write(row_data.describe())

# Model Training Page
elif selection == "Model Training":
    st.title("Model Training")
    data = row_data.dropna(axis=0)
    features = ['tahun', 'jarak_tempuh', 'pajak', 'mpg', 'ukuran_mesin']
    x = data[features]
    y = data['harga']
    train_x, test_x, train_y, test_y = train_test_split(x, y, random_state=40)

    neighbors = st.slider("Select number of neighbors for KNN", 1, 50, 40)
    model = KNeighborsRegressor(n_neighbors=neighbors)
    model.fit(train_x, train_y)

    acc1 = model.score(test_x, test_y)
    test_predict = model.predict(test_x)
    score = mse(test_predict, test_y)

    st.write(f"**MSE:** {score}")
    st.write(f"**Accuracy:** {acc1 * 100:.2f}%")

# Prediction Page
elif selection == "Prediction":
    st.title("Prediction")
    tahun = st.number_input("Year of Manufacture", 2000, 2025, 2012)
    jarak_tempuh = st.number_input("Distance Travelled (in km)", 0, 500000, 12000)
    pajak = st.number_input("Tax (in GBP)", 0, 50000, 1000)
    mpg = st.number_input("Miles Per Gallon (mpg)", 0.0, 100.0, 33.0)
    ukuran_mesin = st.number_input("Engine Size (in liters)", 0.0, 10.0, 2.3)

    row_input = np.array([[tahun, jarak_tempuh, pajak, mpg, ukuran_mesin]])
    neighbors = st.slider("Select number of neighbors for KNN (for prediction)", 1, 50, 28)
    new_model = KNeighborsRegressor(n_neighbors=neighbors)
    new_model.fit(train_x, train_y)

    prediction = new_model.predict(row_input)
    st.write(f"### Predicted Price: £{prediction[0]:.2f}")
    st.write(f"### Predicted Price in IDR (in million): Rp {prediction[0] * 19110 * 1e-6:.2f} Juta")

    # Model prediction
neighbors = st.slider("Select number of neighbors for KNN (for prediction)", 1, 50, 28)
new_model = KNeighborsRegressor(n_neighbors=neighbors)
new_model.fit(train_x, train_y)

prediction = new_model.predict(row_input)
st.write(f"### Predicted Price: £{prediction[0]:.2f}")

st.write(f"### Predicted Price in IDR (in million): Rp {prediction[0] * 19110 * 1e-6:.2f} Juta")

