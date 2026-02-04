# HelperFunctions/validation_rules.py

# Validation rules: column name → function that returns True if valid
validation_rules = {
    "SeniorCitizen": lambda x: x.isin([0, 1]).all(),
    "tenure": lambda x: (x >= 0).all(),
    "MonthlyCharges": lambda x: (x > 0).all(),
    "TotalCharges": lambda x: (x > 0).all(),
    "PaperlessBilling": lambda x: x.isin(["Yes", "No"]).all(),
    "StreamingMovies": lambda x: x.isin(["No", "Yes", "No internet service"]).all(),
    "StreamingTV": lambda x: x.isin(["No", "Yes", "No internet service"]).all(),
    "TechSupport": lambda x: x.isin(["No", "Yes", "No internet service"]).all(),
    "OnlineBackup": lambda x: x.isin(["No", "Yes", "No internet service"]).all(),
    "OnlineSecurity": lambda x: x.isin(["No", "Yes", "No internet service"]).all(),
    "InternetService": lambda x: x.isin(['DSL', 'Fiber optic', 'No']).all(),
    "MultipleLines": lambda x: x.isin(['No phone service', 'No', 'Yes']).all(),
    "PhoneService": lambda x: x.isin(['No', 'Yes']).all(),
    "Dependents": lambda x: x.isin(['No', 'Yes']).all(),
    "Partner": lambda x: x.isin(['No', 'Yes']).all(),
    "gender": lambda x: x.isin(['Female', 'Male']).all(),
    "DeviceProtection": lambda x: x.isin(["No", "Yes", "No internet service"]).all(),
    "Churn": lambda x: x.isin(['No', 'Yes']).all()
}
