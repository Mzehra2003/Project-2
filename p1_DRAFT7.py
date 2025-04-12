#----------------------------------------------
#DRAFT 7: RANDOM FOREST MODEL
#----------------------------------------------

#BLOCK 1: SIMULATING THE DATA

import pandas as pd

# Show all columns when printing a DataFrame
pd.set_option('display.max_columns', None)

# Optionally, increase the overall display width so columns do not wrap
pd.set_option('display.width', 1000)

import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Simulating data
n_samples = 6258  # Number of samples, reset to reflect larger and messier datasets that are more common in MnA data

# The thresholds for the "high risk classification" had to be changed from:
# 1) From the tutorials numbers to test out the model with the revised formula
# 2)Then to experimenting with higher thresholds with the revised number of samples since the Confusion Matrix was inaccurate and not showing every label
# 3) In draft 5, changed number of samples to 487 which made the model run smoothly, but the datapoints were not reflective of real MnA data and wasn't sufficient enough for the k-means analysis
# 4) Moved onto changing the dataset 6000 which resulted in 13 missed deals, and so I revised the danger thresholds as well as added some extra calculations again to make the risk detection more reflective of industry standards


# Generate random data for the features
# Change: needed to add irregular parameters to reflect real time data:
revenue = np.random.normal(100, 25, n_samples)  # Normal distribution for revenue (mean=100, std=25)
costs = np.random.normal(85, 22, n_samples)  # Normal distribution for costs (mean=85, std=22, not an exactly half split as compared to the previous draft)
market_volatility = np.random.normal(0.18, 0.05, n_samples)  # Market volatility simulated as a percentage, mean =18%, s.d = 5%
debt_ratio = np.random.normal(1.7, 0.6, n_samples)  # Revised debt-to-equity ratio (normal distribution)to widen variance to reflect actual MnA debt ratios
current_assets = np.random.normal(90, 20, n_samples) # mean=90, std= 20
current_liabilities = np.random.normal(55, 14, n_samples) # mean= 40, std=14
liquidity = current_assets / current_liabilities # Current ratio 



# Creating a high-risk classification based on revenue, costs, and volatility
# Unlike the one in draft 4, debt ratio is to be included in the risk formula:

#First, we define "danger" thresholds for each risk factor
leverage_threshold = 2.0    # Debt-to-equity ratio above 2.0 is usually considered high risk
liquidity_threshold = 1.2   # Current ratio below 1 can indicate possible liquidity failure
profitability_threshold = 0.10  # Minimum acceptable profit margin, below 5% is danger

#Then, we define the formula for "profit margin" as it is needed in our revised risk formula:
profit_margin = (revenue - costs) / revenue

# Here, we also apply a volatility multiplier to leverage, to simulate risk amplification in unstable markets, similar to stress testing approaches used in financial modeling:
leverage_stress = debt_ratio * (1 + market_volatility) # Since the market volatility is expressed as a percentage, this is the risk adjusted leverage

#Here is our revised risk formula now:
# The risk formula below covers both the class debt ratio, as well as the "leverage stress" stress test
high_risk = (debt_ratio &gt; leverage_threshold) | (leverage_stress &gt; leverage_threshold) | (liquidity &lt; liquidity_threshold) | (profit_margin &lt; profitability_threshold)

high_risk = high_risk.astype(int)  # Convert boolean to 1 or 0

# Create a DataFrame
data = pd.DataFrame({
    'revenue': revenue,
    'costs': costs,
    'market_volatility': market_volatility,
    'debt_ratio': debt_ratio,
    'current_assets': current_assets,
    'current_liabilities': current_liabilities,
    'liquidity': liquidity,
    'profit_margin': profit_margin,
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Scaling the features for better performance of the model (important for RF and other ML models)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit and transform the training data
X_test_scaled = scaler.transform(X_test)  # Only transform the test data

# Check the first few rows of the scaled data (optional)
print(X_train_scaled[:5])



#BLOCK 2.1: K-MEANS CLUSTERING:

from sklearn.cluster import KMeans

# Use all scaled features for clustering (combine train + test for full unsupervised learning)
X_scaled_full = scaler.transform(X)  # Use the full dataset (not split into train/test) for K-means

# Decide on number of clusters (let's start with 3 for business interpretability)
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(X_scaled_full)

# Add cluster labels to your original DataFrame for easy analysis
data['cluster'] = cluster_labels

# View a quick count of each cluster's size
print("Number of deals in each cluster:\n", data['cluster'].value_counts())

# OPTIONAL: See the mean value of each feature in each cluster (to understand what drives each group)
print("\nCluster feature means:")
print(data.groupby('cluster').mean(numeric_only=True))

# See risk label breakdown by cluster
print("\nHigh risk counts per cluster:")
print(data.groupby('cluster')['high_risk'].value_counts())

# Calculate percent of high risk in each cluster for business insight
print("\nPercentage of high risk deals in each cluster:")
print(data.groupby('cluster')['high_risk'].mean().apply(lambda x: f"{x:.2%}"))


# BLOCK 2.2: VISUALISING THE CLUSTERS USING PCA:

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Reduce to 2D for plotting
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled_full)

plt.figure(figsize=(8,6))
for i in range(n_clusters):
    plt.scatter(X_pca[cluster_labels == i, 0], X_pca[cluster_labels == i, 1], label=f'Cluster {i}', alpha=0.6)
plt.title('K-means Clusters (PCA 2D Projection)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()
plt.show()



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

#This is also optional, but for a better business interpretation, we can add a "Business Insight" section by taking the false negatives from the conf matrix
# We use an if-else statement to print the interpretation:
if conf_matrix.shape == (2, 2):  # Makes sure we have both high and low risk companies
    false_negatives = conf_matrix[1,0]  # High-risk companies we missed
    print(f"Business insight: We missed {false_negatives} high-risk companies - damage control needed ASAP pls!")
else:
    print("Business Insight: Looks like smooth sailing! - most companies classified correctly")


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
initial_growth_rate = 0.056  # Economic growth rate (5.6%)
initial_inflation_rate = 0.023  # Inflation rate (2.3%)
initial_regulatory_impact = 0.013  # Simulated regulatory change impact (1.3%)



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
        
        # Simulate Economic Growth Rate (mean=5.6%, volatility=2.7%)
        growth_rate = np.random.normal(initial_growth_rate, 0.02)
        
        # Simulate Inflation Rate (mean=2.3%, volatility=1.7%)
        inflation_rate = np.random.normal(initial_inflation_rate, 0.017)
        
        # Simulate Regulatory Changes (mean=1.3%, volatility=0.5%)
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
