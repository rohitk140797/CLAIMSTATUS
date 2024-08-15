import streamlit as st 
import joblib
import numpy as np

# Load your trained model
reloaded_joblib = joblib.load('claim_status_joblib')

# Title of the app
st.title("Claim Status Prediction")

# Sidebar for user input
st.sidebar.header("Input Parameters")

# Create input fields for each feature
subscription_length = st.sidebar.number_input("Subscription Length", min_value=0, max_value=100, value=12)
vehicle_age = st.sidebar.number_input("Vehicle Age", min_value=0, max_value=20, value=5)
customer_age = st.sidebar.number_input("Customer Age", min_value=18, max_value=100, value=30)
region_code = st.sidebar.number_input("Region Code", min_value=0, max_value=50, value=10)
region_density = st.sidebar.number_input("Region Density", min_value=0, max_value=10, value=5)
segment = st.sidebar.number_input("Segment", min_value=0, max_value=10, value=3)
model = st.sidebar.number_input("Model", min_value=0, max_value=100, value=50)
fuel_type = st.sidebar.number_input("Fuel Type", min_value=0, max_value=10, value=1)
max_torque = st.sidebar.number_input("Max Torque", min_value=0, max_value=1000, value=200)
max_power = st.sidebar.number_input("Max Power", min_value=0, max_value=1000, value=150)
engine_type = st.sidebar.number_input("Engine Type", min_value=0, max_value=10, value=3)
airbags = st.sidebar.number_input("Airbags", min_value=0, max_value=10, value=2)
is_esc = st.sidebar.number_input("Is ESC", min_value=0, max_value=1, value=1)
is_adjustable_steering = st.sidebar.number_input("Is Adjustable Steering", min_value=0, max_value=1, value=1)
is_tpms = st.sidebar.number_input("Is TPMS", min_value=0, max_value=1, value=1)
is_parking_sensors = st.sidebar.number_input("Is Parking Sensors", min_value=0, max_value=1, value=1)
is_parking_camera = st.sidebar.number_input("Is Parking Camera", min_value=0, max_value=1, value=1)
rear_brakes_type = st.sidebar.number_input("Rear Brakes Type", min_value=0, max_value=10, value=3)
displacement = st.sidebar.number_input("Displacement", min_value=0, max_value=10000, value=1500)
cylinder = st.sidebar.number_input("Cylinder", min_value=0, max_value=12, value=4)
transmission_type = st.sidebar.number_input("Transmission Type", min_value=0, max_value=10, value=2)
steering_type = st.sidebar.number_input("Steering Type", min_value=0, max_value=10, value=3)
turning_radius = st.sidebar.number_input("Turning Radius", min_value=0, max_value=20, value=5)
length = st.sidebar.number_input("Length", min_value=0, max_value=10000, value=4500)
width = st.sidebar.number_input("Width", min_value=0, max_value=3000, value=1800)
gross_weight = st.sidebar.number_input("Gross Weight", min_value=0, max_value=10000, value=1500)
is_front_fog_lights = st.sidebar.number_input("Is Front Fog Lights", min_value=0, max_value=1, value=1)
is_rear_window_wiper = st.sidebar.number_input("Is Rear Window Wiper", min_value=0, max_value=1, value=1)
is_rear_window_washer = st.sidebar.number_input("Is Rear Window Washer", min_value=0, max_value=1, value=1)
is_rear_window_defogger = st.sidebar.number_input("Is Rear Window Defogger", min_value=0, max_value=1, value=1)
is_brake_assist = st.sidebar.number_input("Is Brake Assist", min_value=0, max_value=1, value=1)
is_power_door_locks = st.sidebar.number_input("Is Power Door Locks", min_value=0, max_value=1, value=1)
is_central_locking = st.sidebar.number_input("Is Central Locking", min_value=0, max_value=1, value=1)
is_power_steering = st.sidebar.number_input("Is Power Steering", min_value=0, max_value=1, value=1)
is_driver_seat_height_adjustable = st.sidebar.number_input("Is Driver Seat Height Adjustable", min_value=0, max_value=1, value=1)
is_day_night_rear_view_mirror = st.sidebar.number_input("Is Day Night Rear View Mirror", min_value=0, max_value=1, value=1)
is_ecw = st.sidebar.number_input("Is ECW", min_value=0, max_value=1, value=1)
is_speed_alert = st.sidebar.number_input("Is Speed Alert", min_value=0, max_value=1, value=1)
ncap_rating = st.sidebar.number_input("NCAP Rating", min_value=0, max_value=5, value=4)

# Collecting all the features into a single array
input_features = np.array([[subscription_length, vehicle_age, customer_age, region_code,
                            region_density, segment, model, fuel_type, max_torque,
                            max_power, engine_type, airbags, is_esc,
                            is_adjustable_steering, is_tpms, is_parking_sensors,
                            is_parking_camera, rear_brakes_type, displacement, cylinder,
                            transmission_type, steering_type, turning_radius, length,
                            width, gross_weight, is_front_fog_lights, is_rear_window_wiper,
                            is_rear_window_washer, is_rear_window_defogger, is_brake_assist,
                            is_power_door_locks, is_central_locking, is_power_steering,
                            is_driver_seat_height_adjustable, is_day_night_rear_view_mirror,
                            is_ecw, is_speed_alert, ncap_rating]])

# Predict using the loaded model
if st.sidebar.button("Predict"):
    prediction = reloaded_joblib.predict(input_features)
    st.write(f"The predicted claim status is: {prediction[0]}")
