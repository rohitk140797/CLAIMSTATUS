import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

# Custom CSS for styling
st.markdown(
    """
    <style>
    .main {
        background-color: #f0f2f6;  /* Light grey background */
    }
    h1 {
        color: #ff6347;  /* Tomato color for the title */
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;  /* Light color for the sidebar */
    }
    .prediction {
        font-size: 24px;
        font-weight: bold;
        color: #4caf50;  /* Green color for the prediction text */
    }
    .chart-header {
        font-size: 20px;
        font-weight: bold;
        color: black;  /* Black and bold for chart headers */
    }
    </style>
    """, 
    unsafe_allow_html=True
)

# Load your trained model
reloaded_joblib = joblib.load('claim_status_joblib_new')

# Title of the app
st.title("Claim Status Prediction")

# Sidebar for user input
st.sidebar.header("Input Parameters")

# Create input fields for each feature
subscription_length = st.sidebar.number_input("Subscription Length", min_value=0, max_value=100, value=12)
vehicle_age = st.sidebar.number_input("Vehicle Age", min_value=0, max_value=20, value=5)
customer_age = st.sidebar.number_input("Customer Age", min_value=18, max_value=100, value=30)
region_code = st.sidebar.number_input("Region Code", min_value=0, max_value=50, value=10)

# Collecting all the features into a single array
input_features = np.array([[subscription_length, vehicle_age, customer_age, region_code]])

# Predict using the loaded model
if st.sidebar.button("Predict"):
    prediction = reloaded_joblib.predict(input_features)
    
    # Display the prediction in bold and green
    st.markdown(f"<p class='prediction'>The predicted claim status is: {prediction[0]}</p>", unsafe_allow_html=True)

# Add a black and bold chart header for the histogram
st.markdown("<p class='chart-header'>Customer Age Distribution</p>", unsafe_allow_html=True)

# Dummy data for age distribution
age_data = np.random.randint(18, 80, 100)

fig, ax = plt.subplots()
sns.histplot(age_data, bins=10, kde=True, ax=ax)
ax.set_title("Customer Age Histogram")
ax.set_xlabel("Age")
ax.set_ylabel("Frequency")
st.pyplot(fig)

# Add a black and bold chart header for the pie chart
st.markdown("<p class='chart-header'>Vehicle Age Distribution Pie Chart</p>", unsafe_allow_html=True)

# Dummy data for vehicle age distribution
vehicle_age_data = [25, 15, 35, 10, 15]
labels = ['0-5 years', '6-10 years', '11-15 years', '16-20 years', '21+ years']

fig_pie = go.Figure(data=[go.Pie(labels=labels, values=vehicle_age_data, hole=0.3)])
fig_pie.update_traces(hoverinfo='label+percent', textinfo='value+percent', textfont_size=12)

st.plotly_chart(fig_pie)
