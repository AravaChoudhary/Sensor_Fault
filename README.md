# Machine Learning Sensor Fault Project

## Table of Contents
- [Introduction](#Introduction)
- [Problem Statement](#problem-statement)
- [Machine Learning Workflow](#Machine-Learning-Workflow)
- [Model Deployment](#Model-Deployment)
- [Output (Target)](#Output-Target)
- [Machine Learning Task](#Machine-Learning-Task)
- [Real-Time Prediction](#Real-Time-Prediction)
- [Docker & AWS](#Docker-and-AWS)
- [Feedback and Continuous Improvement](#Feedback-and-Continuous-Improvement)
- [Technologies Used](#Technologies-Used)
- [Contribution](#Contribution)



# Introduction

In semiconductor manufacturing, wafers serve as the foundation for integrated circuits and solar cells. The fabrication of these wafers is highly complex, involving multiple steps and numerous sensor readings to ensure quality. The Sensor Fault Detection Project aims to build a machine learning model to classify wafers as “Good” or “Bad” based on sensor data, helping identify faulty wafers in real time, reducing production waste, and improving yield.

# Problem Statement

The objective is to develop a binary classification model that can predict the quality of semiconductor wafers using 590 sensor readings collected during their fabrication process. Each wafer is labeled as either “Good” (1) or “Bad” (-1), and the model’s goal is to accurately classify wafers into these two categories based on sensor data.


# Project Details

	•	Inputs (Features): The dataset consists of 590 sensor readings for each wafer. These readings capture key environmental and process parameters like temperature, pressure, gas flow, and chemical composition during the wafer fabrication process.
	•	Output (Target):
	•	1: Good wafer
	•	-1: Bad wafer
	•	Goal: Accurately predict wafer quality to detect defects early, thereby improving production efficiency and reducing waste.


# Machine Learning Workflow

## 1. Data Preprocessing

		•	Data Cleaning: Handle missing values and remove noise.
		•	Feature Engineering: Extract and select relevant features from sensor data to enhance model 		performance.
		•	Train-Test Split: Divide the data into training and testing sets.

## Model Development

	For the Sensor Fault Detection project, multiple machine learning algorithms were evaluated and fine-tuned for binary classification. Below are the details:

	- Algorithm Selection:The following algorithms were evaluated based on the dataset:
	- Logistic Regression: Used for its simplicity and interpretability.
	- Random Forest Classifier: To capture complex interactions between features.
	- Gradient Boosting Classifier: For handling non-linear relationships effectively.
	- XGBoost Classifier: A powerful boosting algorithm widely used in industry.

	- Model Training:  
		Cross-validation was used to train the models on the training set and validate their performance.

	- Hyperparameter Tuning:  
		Hyperparameter tuning was performed using `GridSearchCV` to optimize model parameters for better accuracy, precision, and recall.


# Model Deployment

Once the model has been trained and validated, it can be deployed into the production environment where real-time data will be fed into it for prediction. and used AWS for this .

# Output (Target):
The target variable is labeled as "Good/Bad," where 1 indicates a "Good" wafer, and -1 indicates a "Bad" wafer.

# Machine Learning Task:
	Type: Supervised Learning (Classification)
	Model Type: Binary Classification Model

# Real-Time Prediction

	•	Sensor Integration: Collect real-time data from sensors installed at various stages of wafer fabrication.
	•	Prediction Engine: The model continuously processes incoming sensor data to predict whether each wafer is likely to be good or bad.
	•	Actionable Insights:
	•	Flag faulty wafers for further inspection or removal.
	•	Trigger adjustments in the fabrication process based on prediction patterns to prevent future defects.

# Feedback and Continuous Improvement

	•	Performance Monitoring: Continuously monitor model performance and compare predictions with actual outcomes to detect any drifts in accuracy.
	•	Model Retraining: Periodically retrain the model using new data to adapt to changes in the fabrication process or sensor behavior.
	•	Anomaly Detection: Implement anomaly detection to identify sensor faults or process anomalies, further improving production efficiency.

#  Docker & AWS

The project uses Docker for containerization and AWS for deployment

	AWS Services used:
		EC2: For hosting the application.
		S3: For storage.
		AWS App Runner: To automate deployment.

	The model was deployed on an EC2 instance, using S3 for data storage and AWS AppRunner for managing continuous integration and delivery. MongoDB Atlas was used for database management, handling sensor data storage and retrieval efficiently.

Continuous Integration/Continuous Delivery (CI/CD)

	The CI/CD pipeline is configured using GitHub Actions to automate code testing and deployment:

	Continuous Integration
	Linting the code
	Running unit tests
	Continuous Delivery:
	Build, tag, and push Docker image to Amazon ECR
	Deploy the Docker container on AWS using EC2


# Technologies Used

	•	Languages: Python
	•	Libraries: Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn, Plotly
	•	Tools: Google Colaboratory, GitHub, Docker (for containerization)
	•	Cloud: Amazon Web Services (EC2, S3, AppRunner)
	•	DataBase: MongoDB Atlas

# Contribution
Feel free to submit issues or pull requests if you’d like to contribute to the project. Contributions are always welcome!