# Sales Prediction for Retail Store Using Machine Learning

## Project Name

**Sales Prediction for Retail Store Using Machine Learning**

## Project Objective

The objective of this project is to build a machine learning model that predicts future retail sales using historical sales data. By analyzing factors such as promotions, holidays, store type, and seasonal trends, the model helps retailers forecast daily, weekly, and monthly sales to support inventory planning, staffing, and marketing decisions.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Jupyter Notebook

## Project Features

- Data preprocessing and cleaning
- Handling missing values
- Exploratory Data Analysis (EDA)
- Feature engineering from date attributes (Year, Month, Week, Day)
- Encoding categorical variables using LabelEncoder
- Feature scaling using StandardScaler
- Model training using Linear Regression
- Model training using XGBoost Regressor
- Model evaluation using MAE, RMSE, and R² Score
- Sales trend visualization using line and bar charts
- Feature Importance Visualization
- Future sales prediction for new data
- Model comparison to identify the best-performing algorithm
- Model saving using Joblib

## Dataset

The project uses the **Rossmann Store Sales Dataset**, which contains historical sales records along with store information, promotions, holidays, customer counts, store type, assortment type, and competition details. The dataset is used to train and evaluate regression models for forecasting future retail sales.

## Project Workflow

1. Import required Python libraries.
2. Load the Rossmann Store Sales dataset.
3. Explore and understand the dataset.
4. Perform data cleaning and handle missing values.
5. Conduct Exploratory Data Analysis (EDA).
6. Extract date-based features such as Year, Month, Week, and Day.
7. Encode categorical variables using LabelEncoder.
8. Split the dataset into training and testing sets.
9. Scale numerical features using StandardScaler (for Linear Regression).
10. Train Linear Regression and XGBoost Regressor models.
11. Evaluate model performance using MAE, RMSE, and R² Score.
12. Compare the performance of both regression models.
13. Visualize actual vs predicted sales and feature importance.
14. Predict future daily, weekly, and monthly sales.
15. Save the trained model using Joblib.

## Outcome

The machine learning models successfully predict future retail sales based on historical sales data and business-related factors. The project identifies the key drivers influencing sales performance and compares multiple regression algorithms to select the best-performing model. This enables retailers to make data-driven decisions for inventory management, workforce planning, promotional campaigns, and overall business growth.
