# **SoH Prediction from EIS Data**

This project aims to predict the State of Health (SoH) of Lithium-Polymer batteries based on Electrochemical Impedance Spectroscopy (EIS) data. The pipeline is structured to be modular and scalable, covering data processing, feature engineering, model training, and inference.

This version supports multiple machine learning models:

* Random Forest (rf)  
* XGBoost (xgb)  
* Support Vector Machine (svm)

## **Project Structure**

The project expects the raw data to be organized as follows:

soh\_prediction/  
│  
├── data/  
│   ├── processed/  
│   └── raw/  
│       └── LiPO 1/  
│           ├── Discharge\_curve/  
│           └── EIS\_Charge\_discharge/  
│  
├── models/  
├── notebooks/  
├── soh/  
│  
├── main.py  
└── requirements.txt

## **How to Use**

### **1\. Setup**

**a. Install Dependencies:**

pip install \-r requirements.txt

b. Place Data:  
Place your LiPO 1 data folder into the data/raw/ directory.

### **2\. Run the Pipeline**

a. Create Features:  
This is a one-time step to process all raw data into a feature table.  
python main.py create\_features

b. Train a Model:  
Use the \--model flag to choose which model to train. The trained model will be saved in the models/ directory.  
\# Train a Random Forest model (default)  
python main.py train \--model rf

\# Train an XGBoost model  
python main.py train \--model xgb

\# Train a Support Vector Machine model  
python main.py train \--model svm

c. Run Inference:  
Use the predict command to estimate SoH for a new EIS file. You must specify which trained model to use.  
\# Predict using the trained Random Forest model  
python main.py predict \--model rf \--file\_path "path/to/your/new\_eis\_file.csv" \--soc 0.9

\# Predict using the trained XGBoost model  
python main.py predict \--model xgb \--file\_path "path/to/your/new\_eis\_file.csv" \--soc 0.5

## **Customization**

Modify the soh/config.py file to change paths, the Equivalent Circuit Model (CIRCUIT\_STRING), or to tune the hyperparameters for each model in the MODEL\_CONFIGS dictionary.