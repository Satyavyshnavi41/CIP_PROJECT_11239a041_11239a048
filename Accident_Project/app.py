import streamlit as st
import pandas as pd
import pickle

model = pickle.load(open("model.pkl", "rb"))

st.title("Accident Fatality Risk Predictor")

temp = st.slider("Temperature", 0, 120, 70)
humidity = st.slider("Humidity", 0, 100, 50)
pressure = st.slider("Pressure", 20, 40, 30)
visibility = st.slider("Visibility", 0, 10, 5)
wind = st.slider("Wind Speed", 0, 50, 10)

weather = st.selectbox("Weather", ["Clear", "Rain", "Fog"])
day = st.selectbox("Day/Night", ["Day", "Night"])

weather_map = {"Clear":0, "Rain":1, "Fog":2}
day_map = {"Day":0, "Night":1}

data = pd.DataFrame([{
    'Temperature(F)': temp,
    'Humidity(%)': humidity,
    'Pressure(in)': pressure,
    'Visibility(mi)': visibility,
    'Wind_Speed(mph)': wind,
    'Weather_Condition': weather_map[weather],
    'Sunrise_Sunset': day_map[day]
}])

if st.button("Predict"):
    prediction = model.predict(data)
    if prediction[0] == 1:
        st.error("High Fatality Risk")
    else:
        st.success("Low Risk")