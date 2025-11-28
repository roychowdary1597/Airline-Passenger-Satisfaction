import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

# -------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------
st.set_page_config(
    page_title="Airline Passenger Satisfaction",
    page_icon="✈️",
    layout="wide",
)

# -------------------------------------------------------
# CUSTOM CSS (Premium UI)
# -------------------------------------------------------
st.markdown("""
<style>

body {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    font-family: 'Segoe UI', sans-serif;
}

h1, h2, h3, h4 {
    color: #ffffff !important;
}

.section-title {
    font-size: 1.5rem;
    font-weight: 700;
    margin-top: 30px;
    color: #38bdf8;
}

.card {
    background: rgba(255,255,255,0.08);
    padding: 25px;
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.1);
    backdrop-filter: blur(10px);
    box-shadow: 0 8px 25px rgba(0,0,0,0.25);
}

.stButton > button {
    background: linear-gradient(90deg, #06b6d4, #3b82f6);
    color: white;
    padding: 12px 30px;
    border-radius: 12px;
    border: none;
    font-size: 1.1rem;
    font-weight: 600;
    transition: 0.3s;
}

.stButton > button:hover {
    transform: scale(1.05);
    background: linear-gradient(90deg, #0ea5e9, #2563eb);
}

.result-card {
    margin-top: 25px;
    background: rgba(255,255,255,0.1);
    padding: 30px;
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.15);
    animation: fadeIn 0.8s ease-in-out;
}

@keyframes fadeIn {
    0% {opacity: 0; transform: translateY(10px);}
    100% {opacity: 1; transform: translateY(0);}
}

label, .stSlider, .stSelectbox {
    color: #e2e8f0 !important;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# LOAD MODEL + PREPROCESSOR
# -------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model("ann_model1.h5")
    bundle = joblib.load("preprocessor1.pkl")

    if isinstance(bundle, dict) and "preprocessor" in bundle:
        return model, bundle["preprocessor"]

    return model, bundle


def get_satisfaction_probability(pred):
    arr = np.array(pred)
    if arr.ndim == 2 and arr.shape[1] == 2:
        return float(arr[0, 1])  # softmax
    return float(arr.ravel()[0])  # sigmoid


try:
    model, preprocessor = load_artifacts()
except:
    st.error("❌ Could not load model or preprocessor. Ensure files are in same directory.")
    st.stop()


# -------------------------------------------------------
# HEADER
# -------------------------------------------------------
st.markdown("<h1 style='text-align:center;'>✈️ Airline Passenger Satisfaction — Pro Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#cbd5e1;'>AI-driven real-time satisfaction prediction powered by ANN</p>", unsafe_allow_html=True)


# -------------------------------------------------------
# INPUT FORM UI
# -------------------------------------------------------
st.markdown("<div class='section-title'>Passenger Information</div>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1,1,1])

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age", 0, 120, 25)
    customer_type = st.selectbox("Customer Type", ["Loyal Customer", "disloyal Customer"])

with col2:
    travel_type = st.selectbox("Travel Type", ["Business travel", "Personal Travel"])
    flight_class = st.selectbox("Class", ["Eco", "Eco Plus", "Business"])
    distance = st.number_input("Flight Distance", min_value=0.0, value=800.0, step=10.0)

with col3:
    departure_delay_minutes = st.number_input("Departure Delay (min)", 0, 300, 0)
    arrival_delay_minutes = st.number_input("Arrival Delay (min)", 0, 300, 0)

# -------------------------------------------------------
# SERVICE RATINGS
# -------------------------------------------------------
st.markdown("<div class='section-title'>Service Ratings</div>", unsafe_allow_html=True)

colA, colB, colC = st.columns(3)

with colA:
    wifi_service = st.slider("WiFi Service", 1, 5, 3)
    online_booking_service = st.slider("Online Booking", 1, 5, 3)
    online_support = st.slider("Online Support", 1, 5, 3)
    online_boarding = st.slider("Online Boarding", 1, 5, 3)

with colB:
    gate = st.slider("Gate Convenience", 1, 5, 3)
    seat_comfort = st.slider("Seat Comfort", 1, 5, 3)
    food_drink = st.slider("Food & Drink", 1, 5, 3)
    checkin_service = st.slider("Check-in Service", 1, 5, 3)

with colC:
    dep_val_time_convenient = st.slider("Timing Convenience", 1, 5, 3)
    onboard_service = st.slider("On-board Service", 1, 5, 3)
    baggage_handling = st.slider("Baggage Handling", 1, 5, 3)
    leg_room_service = st.slider("Leg Room", 1, 5, 3)
    cleanliness = st.slider("Cleanliness", 1, 5, 3)
    entertainment = st.slider("Entertainment", 1, 5, 3)

# -------------------------------------------------------
# PREDICT BUTTON
# -------------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)
predict_btn = st.button("Predict Satisfaction")

# -------------------------------------------------------
# PREDICTION LOGIC
# -------------------------------------------------------
if predict_btn:

    input_data = pd.DataFrame([{
        "gender": gender,
        "age": age,
        "customer_type": customer_type,
        "travel_type": travel_type,
        "class": flight_class,
        "distance": distance,
        "wifi_service": wifi_service,
        "online_booking_service": online_booking_service,
        "online_support": online_support,
        "online_boarding": online_boarding,
        "gate": gate,
        "seat_comfort": seat_comfort,
        "food_drink": food_drink,
        "checkin_service": checkin_service,
        "dep_val_time_convenient": dep_val_time_convenient,
        "onboard_service": onboard_service,
        "baggage_handling": baggage_handling,
        "leg_room_service": leg_room_service,
        "cleanliness": cleanliness,
        "entertainment": entertainment,
        "departure_delay_minutes": departure_delay_minutes,
        "arrival_delay_minutes": arrival_delay_minutes
    }])

    try:
        X_new = preprocessor.transform(input_data)
        pred = model.predict(X_new)
        prob_sat = get_satisfaction_probability(pred)

        label = "Satisfied 😊" if prob_sat >= 0.5 else "Dissatisfied 😞"
        confidence = prob_sat if prob_sat >= 0.5 else 1 - prob_sat

        # Result card
        st.markdown(f"""
        <div class='result-card'>
            <h2 style='color:white;'>Prediction Result</h2>
            <h3 style='color:#38bdf8;'>{label}</h3>
            <p style='color:#e2e8f0;font-size:1.1rem;'>Confidence: <strong>{confidence*100:.2f}%</strong></p>
        </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error("❌ Prediction error. Please review inputs.")
        st.exception(e)
