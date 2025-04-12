#----------------------------------------------
#DRAFT 4: RANDOM FOREST MODEL
#----------------------------------------------

#BLOCK 1: SIMULATING THE DATA

import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Simulating data
n_samples = 500  # Number of samples

# Generate random data for the features
revenue = np.random.normal(100, 20, n_samples)  # Normal distribution for revenue (mean=100, std=20)
costs = np.random.normal(50, 10, n_samples)  # Normal distribution for costs (mean=50, std=10)
market_volatility = np.random.randint(1, 11, n_samples)  # Random market volatility score between 1-10
debt_ratio = np.random.normal(0.5, 0.1, n_samples)  # Debt-to-equity ratio (normal distribution)

# Creating a high-risk classification based on revenue, costs, and volatility
high_risk = (revenue / costs) * market_volatility > 10  # Simple logic for risk, should be validated against company standards in the next code classification
high_risk = high_risk.astype(int)  # Convert boolean to 1 or 0
# Create a DataFrame
data = pd.DataFrame({
    'revenue': revenue,
    'costs': costs,
    'market_volatility': market_volatility,
    'debt_ratio': debt_ratio,
    'high_risk': high_risk  # Target variable
})
# Show the first few rows of the simulated data
print(data.head())



#BLOCK 2: CLEANING AND PREPROCESSING THE DATA:

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Check for missing values (to ensure the simulated data is complete)
print(data.isnull().sum())  # No missing values expected in simulated data

# Split the data into features (X) and target (y)
X = data.drop(columns=['high_risk'])  # Features (all columns except target)
y = data['high_risk']  # Target variable (high_risk)

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling the features for better performance of the model (important for RF and other ML models)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit and transform the training data
X_test_scaled = scaler.transform(X_test)  # Only transform the test data

# Check the first few rows of the scaled data (optional)
print(X_train_scaled[:5])



#BLOCK 3.1: TRAINING THE RANDOM FOREST MODEL:

from sklearn.ensemble import RandomForestClassifier

# Initialize the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model with the training data
rf_model.fit(X_train_scaled, y_train)

# Predictions on the test set
y_pred = rf_model.predict(X_test_scaled)
print(f"Accuracy: {rf_model.score(X_test_scaled, y_test)}")


#BLOCK 3.2: EVALUATING THE MODEL'S PERFORMANCE USING "ACCURACY", "CONFUSION MATRIX" AND "CLASSIFICATION REPORT"

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Assign titles to be printed alongside each metric for presentation

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}') 

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print(f'Confusion Matrix:\n{conf_matrix}')

# Classification Report
class_report = classification_report(y_test, y_pred)
print(f'Classification Report:\n{class_report}')



#BLOCK 4: VISUALISING HOW MUCH EACH FEATURE CONTRIBUTES TO THE FINAL DECISION MAKING PROCESS,AND USING THE "CONFUSION MATRIX" TO FULLY UNDERSTAND THE MODEL'S PERFORMANCE:
import matplotlib.pyplot as plt
import seaborn as sns

# Plotting the feature importances
feature_importances = rf_model.feature_importances_
features = X.columns

plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances, y=features)
plt.title('Feature Importance in Random Forest Model')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.show()

# Plotting the confusion matrix
plt.figure(figsize=(6, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()




#--------------------------------------------------
# MONTE-CARLO SIMULATION
#--------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# BLOCK 1: SETTING INITIAL VALUES FOR THE SIMULATION:

# Number of simulations (Keep it at 1000 for example)
n_simulations = 1000  

# Number of years to simulate (forecasting for 5 years)
years = 5

# Initial values (base values)
initial_revenue = 100  # Starting revenue in millions (adjustable)
initial_interest_rate = 0.03  # Starting interest rate (3%)
initial_growth_rate = 0.05  # Economic growth rate (5%)
initial_inflation_rate = 0.02  # Inflation rate (2%)
initial_regulatory_impact = 0.01  # Simulated regulatory change impact (1%)



# BLOCK 2: SIMULATING FUTURE MARKET CONDITIONS FOR EACH YEAR:

# Initialize an array to store the future revenue outcomes for each simulation
simulated_revenues = np.zeros(n_simulations)

# Running simulation
for i in range(n_simulations):
    future_revenue = initial_revenue  # Starting revenue for each simulation

    # Track simulated variables over the years
    for year in range(years):
        # Randomize Market Volatility (based on historical volatility patterns)
        market_volatility = np.random.normal(0.03, 0.01)  # mean=3%, s.d=1%
        
        # Simulate Interest Rate changes (mean=3%, volatility=1%)
        interest_rate = np.random.normal(initial_interest_rate, 0.01)
        
        # Simulate Economic Growth Rate (mean=5%, volatility=2%)
        growth_rate = np.random.normal(initial_growth_rate, 0.02)
        
        # Simulate Inflation Rate (mean=2%, volatility=1%)
        inflation_rate = np.random.normal(initial_inflation_rate, 0.01)
        
        # Simulate Regulatory Changes (mean=1%, volatility=0.5%)
        regulatory_impact = np.random.normal(initial_regulatory_impact, 0.005)
        
        # Apply these factors to simulate future revenue
        # Formula: Future Revenue = Previous Revenue * (1 + Growth Rate + Market Volatility - Inflation Rate + Regulatory Impact)
        future_revenue *= (1 + growth_rate + market_volatility - inflation_rate + regulatory_impact)

    # Store the final simulated revenue for this iteration
    simulated_revenues[i] = future_revenue



# BLOCK 3.1: VISUALISING RESULTS:

# Plotting the distribution of the simulated revenues
plt.figure(figsize=(10, 6))
plt.hist(simulated_revenues, bins=50, edgecolor='k', alpha=0.7)
plt.title('Monte Carlo Simulation of Future Revenue (5 Years)')
plt.xlabel('Projected Revenue (in Millions)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()



# Block 3.2: EXTRACTING AND ANALYSING INSIGHTS FROM RESULTS:

# Calculate the mean, standard deviation, and percentiles (for a range of predictions)
mean_revenue = np.mean(simulated_revenues)
std_dev_revenue = np.std(simulated_revenues)
p10 = np.percentile(simulated_revenues, 10)
p90 = np.percentile(simulated_revenues, 90)

print(f"Mean Projected Revenue: {mean_revenue:.2f} million")
print(f"Standard Deviation of Revenue: {std_dev_revenue:.2f} million")
print(f"10th Percentile (P10): {p10:.2f} million")
print(f"90th Percentile (P90): {p90:.2f} million")


