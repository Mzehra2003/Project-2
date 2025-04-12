# ------------------------
# DRAFT 1: RF MODEL
# ------------------------

# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# BLOCK 1: SIMULATING DATA (revenue, costs, market volatility, debt ratio)

np.random.seed(42)  #Set random seed for reproducibility
n_samples = 500  # Simulating data

# Generate random data for features
revenue = np.random.normal(100, 20, n_samples)  # Simulating revenue
costs = np.random.normal(50, 10, n_samples)    # Simulating costs
market_volatility = np.random.randint(1, 11, n_samples)  # Random market volatility
debt_ratio = np.random.normal(0.5, 0.1, n_samples)  # Debt-to-equity ratio

# Creating a high-risk classification based on revenue, costs, and volatility
high_risk = (revenue / costs) * market_volatility > 10 # Simple logic for risk classification
high_risk = high_risk.astype(int) # Convert boolean to 1 or 0

# Create DataFrame
data = pd.DataFrame({
    'revenue': revenue,
    'costs': costs,
    'market_volatility': market_volatility,
    'debt_ratio': debt_ratio,
    'high_risk': high_risk  # Target variable
})

# BLOCK 2: SPLITTING AND TRAINING THE DATA

# Split into features (X) and target (y)
X = data.drop(columns=['high_risk'])  # Features (all columns except target)
y = data['high_risk']  # Target variable (high_risk)

# Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train RF model 
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)  # Initialize the Random Forest model
rf_model.fit(X_train, y_train)  # Train the model with the training data

# Predict and evaluate accuracy
y_pred = rf_model.predict(X_test)
print(f"Accuracy: {rf_model.score(X_test, y_test)}") 

# Confusion matrix and classification report
print(confusion_matrix(y_test, y_pred)) # confusion matrix
print(classification_report(y_test, y_pred))  # classification report


