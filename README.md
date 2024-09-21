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

### What is **joblib**?

**Joblib** is a Python library used for efficiently saving and loading large data objects, such as machine learning models or large datasets. It is particularly useful when you want to **serialize** (save) an object so that you can later **deserialize** (load) it without having to retrain the model every time.

#### Key Points about **joblib**:
- It is faster and more efficient than the default `pickle` module for large arrays or models.
- It’s widely used in machine learning to save models once they have been trained.

### What Does **joblib.dump** Do?

In this specific line of code:
```python
joblib.dump(rf_model_oversampled, 'claim_status_joblib_new')
```

- **`rf_model_oversampled`**: This is the **trained Random Forest model** that you created using oversampled data to address class imbalance.
- **`'claim_status_joblib_new'`**: This is the **file name** under which the model will be saved. The model is saved in a file called `claim_status_joblib_new` for later use.

The function `joblib.dump()` **saves the model** (`rf_model_oversampled`) into a file (`claim_status_joblib_new`) so you can reuse the model without retraining it.

### Why Use **joblib.dump**?

1. **Efficiency**: Once you have trained your model, you don’t want to spend time retraining it every time you use it. By saving the trained model with `joblib.dump`, you can quickly load it later and use it to make predictions.
  
2. **Reusable**: The saved file can be **shared or deployed**. For instance, when deploying the app on Streamlit, you load the model (`joblib.load('claim_status_joblib_new')`) to make predictions without needing to retrain it each time.

### Why Save the Model in This Case?
In your project:
- **`rf_model_oversampled`** is the Random Forest model that was trained using an **oversampling technique** to balance the dataset (handling the class imbalance between insurance claims and non-claims).
  
By saving it to `'claim_status_joblib_new'`, you ensure that the model can be reused:
- In your **Streamlit app** where you load the model to make predictions.
- In **future analyses or deployments**, where you don’t need to retrain the model from scratch.

### Summary of **joblib.dump**:
- **joblib** is used for efficiently saving and loading large objects like machine learning models.
- **`joblib.dump(model, 'filename')`** saves the trained model to a file.
- In your case, **`claim_status_joblib_new`** stores the Random Forest model so it can be quickly loaded and used later.

By saving the model, you avoid retraining and can instantly deploy it in your app or project.



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

Here’s a simplified guide to help you **present** the process of pushing the app to GitHub and running it on Streamlit:

---

### **Objective**:
 Want to deploy an interactive web app that predicts insurance claims using a trained model, and you will host this app for free on Streamlit Cloud. Here’s how to do it in simple steps.

---

### **Step 1: Organize Your Files**

Before pushing your project, you need to make sure you have all the necessary files ready in one folder:

- **`appnew.py`**: This is the Python file for your Streamlit app.
- **`claim_status_joblib_new`**: Your trained model file.
- **`Procfile`**: Tells the platform how to start your app. (Content: `web: sh setup.sh && streamlit run appnew.py`)【15†source】
- **`requirements.txt`**: A list of all the libraries (like `streamlit`, `joblib`, etc.) that your app needs to run【16†source】.
- **`setup.sh`**: Configures Streamlit settings (like making it run in "headless" mode, useful for deployment)【17†source】.

---

### **Step 2: Push the Files to GitHub**

Now, you will create a GitHub repository and push your project files there.

1. **Create a GitHub Repository**:
   - Go to [GitHub](https://github.com) and log in.
   - Click **"New repository"** and give it a name (e.g., “insurance-claim-prediction”).
   - Set it to **Public** so Streamlit can access it.

2. **Push Files to GitHub**:
   - Open your terminal/command prompt on your computer.
   - Navigate to the folder where you have saved all the files using the `cd` command.
   - Run these commands to send your files to GitHub:
     ```bash
     git init  # Initialize Git
     git add .  # Add all files
     git commit -m "Initial commit"
     git remote add origin https://github.com/your-username/insurance-claim-prediction.git
     git push -u origin main  # Push files to GitHub
     
What is Git LFS?
Git LFS is a system for managing large files in Git repositories.
It stores large files (like machine learning models, datasets, etc.) outside the main repository while still keeping references to them in the repository.
     ```
     GitHub has a limit of 100 MB for normal file uploads, and since your claim_status_joblib_new file is 129 MB, using Git LFS is the best way to manage it. It ensures that your large model file doesn’t slow down the repository while still being available when needed.
    
     git lfs track "claim_status_joblib_new"

---

### **Step 3: Deploy the App on Streamlit Cloud**

You will now make your app live using Streamlit Cloud.

1. **Go to Streamlit Cloud**:
   - Visit [Streamlit Cloud](https://share.streamlit.io/) and sign in using GitHub.

2. **Create a New App**:
   - Click **"New app"** and select the repository you created in GitHub.
   - Choose the correct branch (`main`) and the main file (`appnew.py`).

3. **Set Up Your App**:
   - Streamlit will automatically install the necessary libraries (as defined in your `requirements.txt`).
   - The **Procfile** and **setup.sh** will ensure the app runs correctly.

4. **Deploy**:
   - Click **"Deploy"**, and within a few seconds, your app will be live!
   - You’ll get a URL you can share with others.

---

### **Conclusion**

By following these steps, you can easily push your app to GitHub, deploy it on Streamlit Cloud, and share the live link with everyone. The app will predict insurance claims based on user inputs and show some cool visualizations.

---

### **Key Points to Highlight During Presentation**:

- **GitHub**: A platform to store and share your project code.
- **Streamlit Cloud**: A free service to deploy Python apps.
- **`requirements.txt`**: This file lists all the necessary Python packages (like `streamlit`, `joblib`, etc.).
- **`Procfile`** and **`setup.sh`**: These files ensure that your app runs correctly on Streamlit.

Let's dive deeper into **Procfile** and **setup.sh**, explaining their roles in simple terms and how they help in deploying your Streamlit app on platforms like Streamlit Cloud or Heroku.

---

### **Procfile**:  
Think of **Procfile** as an instruction manual that tells the platform (Streamlit Cloud, Heroku, etc.) how to run your application. It ensures your app launches correctly when deployed.

#### **What's Inside the Procfile?**
Your **Procfile** contains the following line:
```bash
web: sh setup.sh && streamlit run appnew.py
```

- **`web:`**: This part tells the platform you are running a web application. It's a standard way to define how web processes should be started.
  
- **`sh setup.sh`**: This command runs the **setup.sh** script (explained below). It configures the environment to make sure Streamlit is set up correctly (like setting the correct port and disabling CORS).

- **`streamlit run appnew.py`**: This command actually starts the Streamlit app using the `appnew.py` file. It’s the core command that tells Streamlit to load and serve your app.

In summary, the **Procfile** tells the platform: 
1. **Run the `setup.sh` script** to configure the server.
2. **Launch the Streamlit app** using `appnew.py`.

---

### **setup.sh**:  
The **setup.sh** file is a **shell script** that helps set up the environment in which your Streamlit app will run. It ensures that the app is accessible and can be run without issues.

#### **What's Inside the setup.sh?**
Your **setup.sh** file contains this:
```bash
mkdir -p ~/.streamlit/  # Create a directory for Streamlit configuration files

echo "\
[server]\n\  # Configure Streamlit server settings
headless = true\n\  # Run in headless mode (no GUI)
port = $PORT\n\  # Use the correct port, dynamically set by the platform (Heroku or Streamlit Cloud)
enableCORS = false\n\  # Disable Cross-Origin Resource Sharing for security flexibility
\n\
" > ~/.streamlit/config.toml  # Write these settings to the config file
```

- **`mkdir -p ~/.streamlit/`**: This command creates a hidden directory `~/.streamlit/` in the environment. This is where the configuration file for Streamlit will be stored.

- **`echo`**: The `echo` command writes a configuration file (`config.toml`) in the `.streamlit` folder. This file contains server settings like:
  - **`headless = true`**: This tells Streamlit to run in "headless mode", which means no graphical user interface (perfect for cloud deployment).
  - **`port = $PORT`**: The `$PORT` is a placeholder. When deployed, the platform assigns a port dynamically (like 8501 or 80), and this ensures Streamlit listens on the correct port.
  - **`enableCORS = false`**: Disabling CORS (Cross-Origin Resource Sharing) ensures that the app can be accessed from various platforms without restrictions.

In summary, **setup.sh**:
1. **Creates a Streamlit configuration folder**.
2. **Writes the necessary configuration** (like setting the port, running in headless mode) into the `config.toml` file.

---

### **Why Are Procfile and setup.sh Important?**
- **Procfile**: It ensures the platform knows exactly **how to start your app**.
- **setup.sh**: It **configures the environment** for Streamlit, ensuring that your app runs smoothly by setting parameters like port and CORS.

---

### **Real-World Example (Step-by-Step Flow)**:
1. When you deploy the app on **Streamlit Cloud**:
   - Streamlit Cloud first looks at the **Procfile**.
   - The **Procfile** tells it to run **setup.sh** and then start **Streamlit**.
   
2. The **setup.sh** script:
   - Creates a `.streamlit/` folder.
   - Sets Streamlit to run in headless mode and listens on the correct port.

3. Once setup is complete, the platform:
   - Runs the app by executing `streamlit run appnew.py`.


