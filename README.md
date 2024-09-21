# Predictive Model for Insurance Claims
Develop a model for insurance claim likelihood, addressing class imbalance.
## Overview
In this project, our task is to develop and evaluate a predictive model that accurately identifies the likelihood of insurance claims, despite the inherent class imbalance in the dataset. The model should maintain high predictive accuracy across both classes, ensuring that insurers can effectively assess risk and allocate resources.

## Dataset
The dataset used for this project contains historical data on insurance claims, including information about the policyholders, their demographics, past claim history, and other relevant features. The dataset is sourced from statso.io.

## Objectives
Develop a predictive model that accurately predicts the likelihood of insurance claims.
Address the class imbalance issue in the dataset to ensure the model maintains high predictive accuracy across both classes (claims vs. non-claims).

Evaluate the performance of the model using appropriate metrics such as precision, recall, F1-score, and ROC-AUC.
Fine-tune the model parameters and explore different algorithms to improve its performance.
Provide insights and recommendations based on the model's predictions to help insurers better assess risk and allocate resources effectively.

## Methodology
We will employ a machine learning approach to develop the predictive model, utilizing techniques such as logistic regression, decision trees, random forests, or gradient boosting. To address the class imbalance issue, we will explore techniques such as oversampling, undersampling, or using class weights during model training. Model evaluation will be conducted using cross-validation and appropriate performance metrics.


This **Streamlit** code creates a web-based app for predicting whether an insurance claim will happen, based on several inputs like **subscription length**, **vehicle age**, **customer age**, and **region code**. It also includes visualizations of age and vehicle age distribution using histograms and pie charts. Here's a breakdown of the code in simple terms with the objective explained:

### **Objective**
The app's main goal is to let users input details about an insurance policy and get a prediction about whether a claim will be made. It also shows charts for age and vehicle age distribution to help users understand data patterns.

### **Detailed Explanation**

1. **Importing Libraries**:
   - `streamlit`: Used to build the interactive web application.
   - `joblib`: Used to load the pre-trained machine learning model.
   - `numpy`: For handling arrays and numeric data.
   - `matplotlib` and `seaborn`: Libraries to create charts (e.g., histograms).
   - `plotly.graph_objects`: For interactive charts like the pie chart.

2. **Custom CSS Styling**:
   - The app includes some custom **CSS** to make it look visually appealing:
     - Light grey background for the main page.
     - Tomato color for the title text.
     - Green color for the prediction result.
     - Bold black text for chart headers.

3. **Loading the Trained Model**:
   - `reloaded_joblib = joblib.load('claim_status_joblib_new')` loads a previously trained machine learning model from a file. This model is used to predict whether an insurance claim will be made based on the user's input.

4. **App Title**:
   - `st.title("Claim Status Prediction")` sets the title of the app.

5. **Sidebar for Input Parameters**:
   - The sidebar (`st.sidebar`) is where the user inputs values for different features of the insurance policy:
     1. **Subscription Length**: How long the insurance policy has been active.
     2. **Vehicle Age**: How old the insured vehicle is.
     3. **Customer Age**: Age of the policyholder.
     4. **Region Code**: A number representing the policyholder's geographic region.

   The user can enter numbers through input fields that have defined minimum and maximum values.

6. **Prediction Button**:
   - If the user clicks the **"Predict"** button in the sidebar, the app collects the input values and passes them to the machine learning model:
     - `input_features = np.array([[subscription_length, vehicle_age, customer_age, region_code]])` creates an array of the input values.
     - `prediction = reloaded_joblib.predict(input_features)` uses the model to make a prediction.
     - The result is displayed on the screen in **green bold text**, which tells the user whether a claim is likely or not: 
       - `st.markdown(f"<p class='prediction'>The predicted claim status is: {prediction[0]}</p>")`.

7. **Customer Age Distribution (Histogram)**:
   - The app includes a chart (histogram) showing the distribution of **customer age**:
     - `np.random.randint(18, 80, 100)` generates dummy data for ages (between 18 and 80).
     - The **seaborn** library is used to plot a histogram that shows how often each age appears in the data. It also adds a **kde** line, which smooths the distribution.
     - The chart title and axis labels are set, and the chart is displayed using `st.pyplot(fig)`.

8. **Vehicle Age Distribution (Pie Chart)**:
   - The app also includes a **pie chart** showing the distribution of vehicle ages:
     - The age categories are divided into ranges: "0-5 years," "6-10 years," and so on.
     - Dummy data is used to represent the number of vehicles in each age group (e.g., 25 vehicles are 0-5 years old).
     - The **plotly** library creates an interactive pie chart with a hover effect, showing the percentage of each group.
     - `st.plotly_chart(fig_pie)` displays the pie chart.

### **Main Purpose of the App**
- **Prediction Tool**: Users can input insurance details and see whether the app predicts an insurance claim or not.
- **Visual Analysis**: The app helps users understand the distribution of data through simple charts like histograms and pie charts.

This makes the app both functional (providing predictions) and informative (helping users visualize trends in the data).

