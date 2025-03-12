# ------------------------
# DRAFT 3: RANDOM FOREST MODEL
# ------------------------

# Import all libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt


#BLOCK 1: SIMULATING THE DATA

# Set random seed for reproducibility
np.random.seed(42)

# Simulating data
n_samples = 500  # Number of samples

# Generate random data 
revenue = np.random.normal(100, 20, n_samples)  # Normal distribution for revenue (mean=100, std=20)
costs = np.random.normal(50, 10, n_samples)  # Normal distribution for costs (mean=50, std=10)
market_volatility = np.random.randint(1, 11, n_samples)  # Random market volatility score between 1-10
debt_ratio = np.random.normal(0.5, 0.1, n_samples)  # Debt-to-equity ratio (normal distribution)

# Creating a high-risk classification based on revenue, costs, and volatility
high_risk = (revenue / costs) * market_volatility > 10  # Simple logic for risk classification
high_risk = high_risk.astype(int)  # Convert boolean to 1 or 0

# Create a DataFrame
data = pd.DataFrame({
    'revenue': revenue,
    'costs': costs,
    'market_volatility': market_volatility,
    'debt_ratio': debt_ratio,
    'high_risk': high_risk  # Target variable
})



# BLOCK 2: CLEANING AND PREPARING THE DATA:

# Checking for missing values (to check if the data is complete)
print(data.isnull().sum())  # No missing values expected in simulated data

# Splitting the data into features (X) and target (y)
X = data.drop(columns=['high_risk'])  # Features (all columns except target)
y = data['high_risk']  # Target variable (high_risk)

# Splitting the data into 80% train and 20% test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling the features 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit and transform the training data
X_test_scaled = scaler.transform(X_test)  # Only transform the test data

# Check the first few rows of the scaled data
print(X_train_scaled[:5])



# BLOCK 3.1: TRAINING THE RF MODEL

rf_model = RandomForestClassifier(n_estimators=100, random_state=42) # Initializing the model
rf_model.fit(X_train_scaled, y_train) # Train the model with the training data

# Prediction and evaluation on the test data
y_pred = rf_model.predict(X_test_scaled)
print(f"Accuracy: {rf_model.score(X_test_scaled, y_test)}")  



# BLOCK 3.2: "ACCURACY", "CONFUSION MATRIX" AND "CLASSIFICATION REPORT" ON THE RF MODEL'S PERFORMANCE
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))




#-----------------------------------------
# MONTE CARLO SIMULATION
#-----------------------------------------

# BLOCK 1: SETTING NUMBER OF SIMULATIONS AND INITIALIZING ARRAY TO STORE PREDICTED REVENUE OUTCOMES economic factors (inflation, interest rates)

n_simulations = 1000 # Number of simulations (1000)

# Initialize an array to store the future revenue outcomes for each simulation
simulated_revenues = np.zeros(n_simulations)



# BLOCK 2: RUN THE SIMULATION

for i in range(n_simulations):
    future_revenue = 100  # Starting revenue for each simulation

    # Simulate market conditions using initial values (growth, market volatility, inflation, interest rates
    growth_rate = np.random.normal(0.05, 0.02)  # Economic growth rate
    volatility = np.random.normal(0.03, 0.01)  # Market volatility
    inflation = np.random.normal(0.02, 0.01)  # Inflation rate
    interest_rate = np.random.normal(0.03, 0.005)  # Interest rate

    # Apply these factors to simulate future revenue
    # Formula: Future Revenue = Previous Revenue * (1 + Growth Rate + Market Volatility - Inflation Rate + Regulatory Impact)
    future_revenue *= (1 + growth_rate + volatility - inflation + interest_rate)

    # Store the simulated revenue for this iteration
    simulated_revenues[i] = future_revenue



# BLOCK 3: VISUALIZING AND PLOTTING THE RESULTS OF THE SIMULATED REVENUE

plt.figure(figsize=(10, 6))
plt.hist(simulated_revenues, bins=50, edgecolor='k', alpha=0.7)
plt.title('Monte Carlo Simulation of Future Revenue')
plt.xlabel('Projected Revenue (Millions)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

