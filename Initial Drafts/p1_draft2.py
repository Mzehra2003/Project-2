# ------------------------
# DRAFT 2: RF MODEL 
# ------------------------

# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

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

# BLOCK 2: SPLITTING, TRAINING AND SCALING THE DATA

# Split into features (X) and target (y)
X = data.drop(columns=['high_risk'])  # Features (all columns except target)
y = data['high_risk']  # Target variable (high_risk)

# Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling the features 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit and transform on training data
X_test_scaled = scaler.transform(X_test)  # Only transform on test data

# BLOCK 3: TRAINING THE RF MODEL

# Training the RF model the data
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)  # Initialize the Random Forest model
rf_model.fit(X_train_scaled, y_train)

# Predict and evaluate on the test set
y_pred_scaled = rf_model.predict(X_test_scaled)
print(f"Accuracy: {rf_model.score(X_test_scaled, y_test)}")  # Improved accuracy after scaling

# BLOCK 4: EVALUATING THE MODEL'S PERFORMANCE USING "CONFUSION MATRIX" AND "CLASSIFICATION REPORT"

# Confusion matrix and classification report
print(confusion_matrix(y_test, y_pred_scaled))
print(classification_report(y_test, y_pred_scaled))

#------------------------------------------------
# MONTE CARLO SIMULATION
#------------------------------------------------

# Simulate Monte Carlo (growth and volatility)
n_simulations = 1000
simulated_revenues = np.zeros(n_simulations)

for i in range(n_simulations):
    future_revenue = 100  # Starting revenue

    # Simulate growth and volatility
    growth_rate = np.random.normal(0.05, 0.02)  # Economic growth
    volatility = np.random.normal(0.03, 0.01)  # Market volatility

    # Apply market conditions
    future_revenue *= (1 + growth_rate + volatility)

    simulated_revenues[i] = future_revenue

# Plot the results of the Monte Carlo simulation
plt.figure(figsize=(10, 6))
plt.hist(simulated_revenues, bins=50, edgecolor='k', alpha=0.7)
plt.title('Monte Carlo Simulation of Future Revenue')
plt.xlabel('Projected Revenue (Millions)')
plt.ylabel('Frequency')
plt.show()

