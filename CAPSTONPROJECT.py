#!/usr/bin/env python
# coding: utf-8

# # Predictive Model for Insurance Claims
# Develop a model for insurance claim likelihood, addressing class imbalance.

# ## Overview
# In this project, our task is to develop and evaluate a predictive model that accurately identifies the likelihood of insurance claims, despite the inherent class imbalance in the dataset. The model should maintain high predictive accuracy across both classes, ensuring that insurers can effectively assess risk and allocate resources.
# 
# ## Dataset
# The dataset used for this project contains historical data on insurance claims, including information about the policyholders, their demographics, past claim history, and other relevant features. The dataset is sourced from statso.io.
# 
# ## Objectives
# Develop a predictive model that accurately predicts the likelihood of insurance claims.
# Address the class imbalance issue in the dataset to ensure the model maintains high predictive accuracy across both classes (claims vs. non-claims).
# 
# Evaluate the performance of the model using appropriate metrics such as precision, recall, F1-score, and ROC-AUC.
# Fine-tune the model parameters and explore different algorithms to improve its performance.
# Provide insights and recommendations based on the model's predictions to help insurers better assess risk and allocate resources effectively.
# 
# ## Methodology
# We will employ a machine learning approach to develop the predictive model, utilizing techniques such as logistic regression, decision trees, random forests, or gradient boosting. To address the class imbalance issue, we will explore techniques such as oversampling, undersampling, or using class weights during model training. Model evaluation will be conducted using cross-validation and appropriate performance metrics.

# In[3]:


import pandas as pd  # Importing the pandas library and using the alias pd for convenient access to its functions
import seaborn as sns  # Importing the seaborn library and using the alias sns for data visualization
import matplotlib.pyplot as plt  # Importing the pyplot module from the matplotlib library and using the alias plt for plotting
import squarify  # Importing the squarify library for creating treemaps

from sklearn.utils import resample  # Importing the resample function from sklearn.utils for data resampling
from sklearn.metrics import (       # Importing various performance metrics from sklearn.metrics
    confusion_matrix,               # For computing confusion matrices
    classification_report,          # For generating classification reports
    accuracy_score,                 # For calculating accuracy scores
    roc_curve, auc                 # For generating ROC curves and calculating AUC
)
from sklearn.ensemble import RandomForestClassifier  # Importing RandomForestClassifier from sklearn.ensemble
from sklearn.preprocessing import LabelEncoder       # Importing LabelEncoder from sklearn.preprocessing
from sklearn.model_selection import train_test_split, cross_val_score  # Importing the train_test_split function from sklearn library for splitting data into training and testing sets
# Importing the RandomForestClassifier class from sklearn library for building a random forest classifier



import warnings  # Importing the warnings module for managing warning outputs
warnings.filterwarnings('ignore')  # Disabling warning outputs

get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")
sns.set(style='whitegrid')  # Setting the style for seaborn plots


# # Project Structure
# 
# ## Loading Data: 
# Load the dataset from a file or source into a DataFrame for analysis.
# 
# ## Data Preprocessing: 
# Handle any missing or erroneous data, perform data type conversions, and clean the dataset for further analysis.
# 
# ## Exploratory Data Analysis (EDA): 
# Explore the dataset to gain insights into its distribution, relationships, and patterns. Visualize key features and relationships between variables.
# 
# ## Oversampling the Minority Class: 
# Implement oversampling techniques to address class imbalance in the dataset.
# 
# ## Feature Engineering: 
# Create new features or transform existing ones to improve the predictive power of the model.
# 
# ## Data Splitting: 
# Split the training dataset into features (x) and the target variable (y) to prepare for model training.
# 
# ## Model Training: 
# Utilize machine learning algorithms to train a predictive model on the training data.
# 
# ## Model Evaluation: 
# Evaluate the trained model's performance using appropriate metrics and techniques.

# # Loading the Data

# In[4]:


# Path to the CSV data file
DATAPATH = 'Insurance claims data.csv' 

# Reading data from the CSV file into a DataFrame using the first column as the index
df = pd.read_csv(DATAPATH)  

# Displaying the first few rows of the DataFrame
df.head()


# In[5]:


# Retrieve the column names of the DataFrame 'df'
column_names = df.columns
# Print the column names
print(column_names)


# # Description of Columns
# 
# **policy_id:** Unique identifier for the insurance policy.
# 
# **subscription_length:** The duration for which the insurance policy is active.
# 
# **vehicle_age:** Age of the vehicle insured.
# 
# **customer_age:** Age of the insurance policyholder.
# 
# **region_code:** The code representing the geographical region of the policyholder.
# 
# **region_density:** Population density of the policyholder’s region.
# 
# **segment:** Segment of the vehicle.
# 
# **model:** The model of the vehicle.
# 
# **fuel_type:** Type of fuel the vehicle uses (e.g., Petrol, Diesel, CNG).
# 
# **max_torque, max_power:** Engine performance characteristics
# 
# **engine_type:** The type of engine.
# 
# **airbags**: The number of airbags in the vehicle.
# 
# **is_esc (Electronic Stability Control), is_adjustable_steering, is_tpms (Tire Pressure Monitoring System)**: Features that enhance vehicle safety.
# 
# **is_parking_sensors, is_parking_camera: Parking aids.**
# 
# **rear_brakes_type:** Type of rear brakes.
# 
# **displacement, cylinder:** Specifications related to the engine size and construction.
# 
# **transmission_type:** Type of transmission.
# 
# **steering_type:** Type of steering mechanism.
# 
# **turning_radius:** Turning radius of the vehicle.
# 
# **length, width, gross_weight:** Dimensions and weight of the vehicle.
# 
# **is_front_fog_lights, is_rear_window_wiper, is_rear_window_washer, is_rear_window_defogger, is_brake_assist, is_power_door_locks**, 
# 
# **is_central_locking, is_power_steering, is_driver_seat_height_adjustable, is_day_night_rear_view_mirror, is_ecw, is_speed_alert:** Various binary indicators for specific vehicle amenities and safety features.
# 
# **ncap_rating:** NCAP safety rating of the vehicle.
# 
# **claim_status:** Indicates whether a claim was made (1) or not (0).
# 

# In[7]:


# This line of code retrieves the shape of the DataFrame 'df'
shape = df.shape
print(shape)


# # Data Preprocessing

# In[8]:


# Displaying concise summary information about the DataFrame, including
# data types, non-null values, and memory usage
df.info()


# In[9]:


# Displaying the data types of each column in the DataFrame
df.dtypes


# In[12]:


# Counting the number of missing values in each column and then counting
# the frequency of those counts to summarize the distribution of missing values
df.isna().sum()


# In[14]:


# Counting the number of duplicated rows in the DataFrame and then counting
# the frequency of those counts to summarize the distribution of duplicated rows
df.duplicated().value_counts()

Hence policy_id is a unique identifier for the insurance policy, this column should be dropped to ensure the uniqueness of each policy doesn't influence the predictive power of the model. Dropping it will help avoid any biases or confusion that might arise from the model mistakenly learning patterns related to these identifiers.
# In[15]:


# Dropping the 'policy_id' column from the DataFrame along the columns axis
df.drop(columns='policy_id', axis=1, inplace=True)


# ## Exploratory Data Analysis

# In[16]:


# Creating a count plot to visualize the distribution of claim status
plt.figure(figsize=(6, 4))
sns.countplot(data=df, y='claim_status', hue='claim_status', palette=['#BBAB8C', '#503C3C'])
plt.xlabel('Count')
plt.ylabel('Claim Status')
plt.title('Distribution of Claim Status')
plt.show()


# In this dataset, there is a significant class imbalance, which may affect the model's ability to learn properly and accurately classify instances.
# 
# To address the class imbalance issue, we will explore resampling techniques during model training. Resampling involves modifying the distribution of the training data to balance the classes.
# 
# **Two common resampling techniques are:**
# 
#    1) **Oversampling**: This involves increasing the number of instances in the minority class by duplicating or creating synthetic      samples. Popular oversampling methods include Random Oversampling and Synthetic Minority Over-sampling Technique (SMOTE).
#    
#   2)   **Undersampling:** This involves reducing the number of instances in the majority class by randomly removing samples. Common    undersampling methods include Random Undersampling and Tomek Links.

# In[17]:


# Selecting the numerical columns from the DataFrame and storing their column names in a variable
numerical_columns = df[['subscription_length', 'vehicle_age', 'customer_age']].columns

# Displaying the column names of the selected numerical columns
numerical_columns


# In[18]:


# plotting distributions of numerical features
plt.figure(figsize=(10, 4))
for i, col in enumerate(numerical_columns, 1):
    plt.subplot(len(numerical_columns) // 3, 3, i)
    sns.kdeplot(df[col], fill = True, color = '#BBAB8C') 
    plt.title(f'Distribution of {" ".join([el.title() for el in col.split("_")])}')
    plt.xlabel(f'{" ".join([el.title() for el in col.split("_")])}')

plt.tight_layout()
plt.show()


# In[19]:


# Creating subplots to visualize the distribution of numerical columns
plt.figure(figsize=(10, 4))
for i, col in enumerate(numerical_columns, 1):
    plt.subplot(1, 3, i)
    sns.histplot(data=df, x=col, color='#BBAB8C', kde=True)
    plt.title(f'Distribution of {" ".join([el.title() for el in col.split("_")])}')
    plt.xlabel(f'{" ".join([el.title() for el in col.split("_")])}')
plt.tight_layout()
plt.show()


# In[20]:


# Creating a heatmap to visualize the correlation matrix of numerical columns
plt.figure(figsize=(12, 10))
sns.heatmap(df.select_dtypes(include='number').corr(), annot=True, cmap=['#A79277', '#D1BB9E', '#EAD8C0', '#FFF2E1', '#FEFAF6'])
plt.show()


# In[21]:


# Specifying a list of categorical columns
categorical_columns = ['region_code', 'segment', 'fuel_type']


# In[22]:


# Creating subplots to visualize the distribution of categorical columns
plt.figure(figsize=(10, 7))
for i, col in enumerate(categorical_columns, 1):
    plt.subplot(1, 3, i)
    sns.countplot(data=df, y=col, order=df[col].value_counts().index, hue=col, color='#C69774')
    plt.title(f'Distribution of {" ".join([el.title() for el in col.split("_")])}')
    plt.xlabel('Count')
    plt.ylabel(f'{" ".join([el.title() for el in col.split("_")])}')
plt.tight_layout()
plt.show()


# # Oversampling the Minority Class

# In[23]:


# Creating subsets of the DataFrame based on the 'claim_status' column:
# - 'minority' contains rows where 'claim_status' is equal to 1
# - 'majority' contains rows where 'claim_status' is equal to 0
minority = df[df['claim_status'] == 1]
majority = df[df['claim_status'] == 0]


# In[24]:


majority.head()


# In[25]:


# Oversampling the minority class to balance the dataset
minority_oversampled = resample(minority, replace=True, n_samples=len(majority), random_state=42)


# In[26]:


# Concatenating the majority class with the oversampled minority class to create a balanced dataset
oversampled_data = pd.concat([majority, minority_oversampled])
oversampled_data.head()


# In[27]:


# Counting the number of occurrences of each class in the 'claim_status' column
oversampled_data['claim_status'].value_counts()


# In[28]:


# Creating subplots to visualize the distribution of numerical columns with KDE plots
plt.figure(figsize=(10, 5))
for i, col in enumerate(numerical_columns, 1):
    plt.subplot(1, 3, i)
    sns.kdeplot(data=oversampled_data, x=col, hue='claim_status', fill=True, palette=['#BBAB8C', '#503C3C'])
    plt.xlabel(f'{" ".join([el.title() for el in col.split("_")])}')
    plt.title(f'Distribution of {" ".join([el.title() for el in col.split("_")])}')
plt.tight_layout()
plt.show()


# # Conclusion
# The distribution of the claim_status target variable after oversampling is as follows:
# 
# **Class 0:** 54844 instances
# 
# **Class 1:** 54844 instances
# 
# ** **
# This balanced distribution indicates that the oversampling technique has effectively addressed the class imbalance issue in the dataset. With an equal number of instances for both classes, we can now proceed to train our predictive model on the balanced dataset, ensuring that both classes are adequately represented and the model can learn from both types of instances effectively.

# # Feature Engineering

# In this section, we'll address the need to preprocess categorical variables in our dataset before training machine learning models. Many machine learning algorithms require numerical input, which means we need to convert categorical variables into a numerical format. One common technique for achieving this is using the LabelEncoder from the scikit-learn library. The LabelEncoder converts categorical labels into numerical labels, allowing us to represent categorical data numerically. We'll initialize a LabelEncoder object and apply it to each column in our dataset where the data type is 'object', effectively transforming categorical variables into a format suitable for model training. Let's dive into the code and preprocess our data using LabelEncoder!

# In[29]:


# Initialize a LabelEncoder
labenc = LabelEncoder()

# Apply LabelEncoder to each column if the column dtype is 'object', else keep the column as is
encoded_data = df.apply(lambda col: labenc.fit_transform(col) if col.dtype == 'object' else col)


# In[30]:


encoded_data.head()


# In[31]:


# Separating features (X) and target variable (y)
X = encoded_data.drop('claim_status', axis=1)
y = encoded_data['claim_status']


# In[32]:


# Instantiate the Random Forest classifier with a random state of 42
rf_model = RandomForestClassifier(random_state=42)

# Fit the model to the training data
rf_model.fit(X, y)


# In[33]:


# Get feature importances from the trained Random Forest model
feature_importance = rf_model.feature_importances_


# In[34]:


# Create a DataFrame to store feature names and their corresponding importances
features_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importance})

# Sort the DataFrame by feature importance in descending order
features_df = features_df.sort_values(by='Importance', ascending=False)

# Display the top 10 most important features
features_df.head(10)


# # Conclusion
# 
# **Based on the feature importance analysis, it is evident that:** 
# 
# **subscription_length:**  0.417150
# ** ** 
# **customer_age:**  0.263397
# ** ** 
# **vehicle_age:**  0.193724
# ** ** 
# **region_density** : 0.059310
# ****
# **region_code:** 0.058201
# ** **
# These results indicate that subscription length, customer age, and vehicle age are the most influential features in predicting insurance claims likelihood. Factors such as region density and region code also contribute to the predictive power, albeit to a lesser extent. Understanding and considering these key features can aid insurers in accurately assessing risk and allocating resources effectively.

# In[35]:


# Separating features (X) and target variable (y) from the oversampled dataset
X_oversampled = oversampled_data.drop('claim_status', axis=1)
y_oversampled = oversampled_data['claim_status']


# In[36]:


# Apply LabelEncoder to each column if the column dtype is 'object', else keep the column as is
X_oversampled_encoded = X_oversampled.apply(lambda col: LabelEncoder().fit_transform(col) if col.dtype == 'object' else col)


# # Data Splitting

# In[37]:


# Splitting the oversampled data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_oversampled_encoded, y_oversampled, test_size=0.4, random_state=42)


# # Model Training

# In[38]:


# Instantiate the Random Forest classifier with a random state of 42
rf_model_oversampled = RandomForestClassifier(random_state=42)

# Fit the model to the training data
rf_model_oversampled.fit(X_train, y_train)


# In[39]:


# Predicting the target variable for the test set using the trained model
y_pred = rf_model_oversampled.predict(X_test)


# In[40]:


# Print the classification report
print(classification_report(y_test, y_pred))


# # Conclusion
# 
# Based on the classification report, our predictive model for insurance claims likelihood assessment demonstrates excellent performance across various evaluation metrics. With a precision of 1.00 for class 0 and 0.95 for class 1, the model achieves high accuracy in predicting both positive and negative instances. Moreover, the recall score of 0.95 for class 0 and 1.00 for class 1 indicates the model's ability to effectively capture true positive instances while minimizing false negatives. The balanced f1-score of 0.97 for both classes further validates the model's robustness in terms of precision and recall. Overall, with an accuracy of 0.97, the model showcases strong predictive capabilities, making it a reliable tool for insurers to assess insurance claims likelihood accurately. This performance underscores the effectiveness of our predictive modeling approach and its potential to contribute to more informed decision-making in the insurance industry.

# # Model Evaluation

# In this section, we'll thoroughly evaluate the performance of our predictive model for insurance claims likelihood assessment. Evaluating the model is essential to understand its effectiveness, generalization ability, and potential limitations. We'll use a variety of evaluation metrics to assess different aspects of the model's performance, including accuracy, precision, recall, F1-score, and ROC-AUC. These metrics provide insights into how well the model predicts both positive and negative instances, its ability to minimize false positives and false negatives, and its overall discriminative power. By comprehensively evaluating the model, we aim to gain a deeper understanding of its strengths and weaknesses, identify areas for improvement, and ultimately make informed decisions about its deployment in real-world applications. Let's delve into the model evaluation process and analyze the performance of our predictive model!

# In[41]:


# Create a DataFrame to store the top 5 most important features and their importances
importance_df = pd.DataFrame({
    'Column': list(map(lambda col: ' '.join(map(str.title, col.split('_'))), X_oversampled_encoded.columns)),
    'Importance': rf_model_oversampled.feature_importances_
}).sort_values(by='Importance', ascending=False).head(5)


# In[42]:


# Set up the figure and plot the treemap
plt.figure(figsize=(5, 5))
squarify.plot(sizes=importance_df['Importance'], label=importance_df['Column'], color='#D1BB9E', edgecolor='white')

# Add title
plt.title('Feature Importance')

# Show the plot
plt.show()


# In[43]:


# Calculate the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix as a heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap=['#D1BB9E', '#A79277'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()


# In[44]:


# Calculate the accuracy score
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)


# In[45]:


y_probs = rf_model_oversampled.predict_proba(X_test)[:, 1]

# Compute ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, color='#D1BB9E', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()


# In[46]:


# Perform k-fold cross-validation
scores = cross_val_score(rf_model_oversampled, X_train, y_train, cv=5, scoring='accuracy')

# Print the mean accuracy and standard deviation of the cross-validation scores
print("Mean Accuracy:", scores.mean())
print("Standard Deviation of Accuracy:", scores.std())


# ## Mean Accuracy: 0.9660092490100773
# 
# ## Standard Deviation of Accuracy: 0.0010790552679904987
# 
# ## Applying Pretrained Model for Prediction
# ** **
# In this section, we'll apply the pretrained model to the original test data to generate predictions for insurance claims likelihood. After training and fine-tuning our model on the training data, it's crucial to evaluate its performance on unseen data to assess its generalization ability. We'll use the imbalanced test data, which contains instances that the model has not seen during training, to make predictions. Subsequently, we'll compare the predicted labels with the actual labels to evaluate the model's accuracy. This analysis will provide insights into how well our model performs in real-world scenarios and its effectiveness in accurately identifying insurance claims likelihood. Let's proceed with applying the pretrained model and evaluating its performance!

# In[48]:


# Create a copy of the original DataFrame
original_encoded = df.copy()

# Initialize a dictionary to store the encoders
encoders = {}

# Iterate over categorical columns and create LabelEncoders
for col in X_oversampled.select_dtypes(include=['object']).columns:
    encoders[col] = LabelEncoder().fit(X_oversampled[col])


# In[49]:


# Printing the dictionary containing fitted LabelEncoders for each categorical column
encoders


# In[50]:


# Transforming categorical columns in the original encoded dataset using fitted LabelEncoders
for col in original_encoded.select_dtypes(include='object').columns:
    if col in encoders:
        original_encoded[col] = encoders[col].transform(original_encoded[col])

# Making predictions on the transformed dataset using the trained model
original_encoded_predictions = rf_model_oversampled.predict(original_encoded.drop('claim_status', axis=1))


# In[51]:


# Create a DataFrame to compare actual and predicted values of the 'claim_status' column
comparison_df = pd.DataFrame({
    'Actual': original_encoded['claim_status'],
    'Predicted': original_encoded_predictions
})

# Display the DataFrame
comparison_df


# In[52]:


# Calculate the counts of correctly and incorrectly classified samples
correctly_classified = (comparison_df['Actual'] == comparison_df['Predicted']).sum()
incorrectly_classified = (comparison_df['Actual'] != comparison_df['Predicted']).sum()

# Store the counts in a list
classification_counts = [correctly_classified, incorrectly_classified]

# Define labels for the counts
labels = ['Correctly Classified', 'Misclassified']

# Create a pie chart
plt.pie(classification_counts, labels=labels, autopct='%1.1f%%', startangle=140, colors=['#E8DFCA', '#D1BB9E'])

# Draw a circle in the center to create a ring
centre_circle = plt.Circle((0,0),0.70,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

# Equal aspect ratio ensures that pie is drawn as a circle
plt.axis('equal')

plt.show()


# In[ ]:





# In[ ]:




