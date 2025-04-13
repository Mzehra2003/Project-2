#===========================================
# Step 1: Descriptive Statistics
#===========================================


# Block 1: Overall Basic Descriptive Stats (See file T2Wrangling_Analysis to navigate to class notes on statistics)

import pandas as pd  # import pandas to work with table-like data
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 0)

# loading the cleaned data from the CSV file into a table called df
df = pd.read_csv("C:\\Users\\MAHA ZEHRA\\Desktop\\fintech_itv_cleaned.csv")

# The first run wasn't showing the app_usage_frequency KPI metrics because the data included in it was "monthly", "weekly" and "daily" instead of numbers,
# So for the categorical KPI to be taken into account as well, I edited this draft to convert the categorical data to numeric as well, as shown below

# Create a mapping from categorical to numeric
frequency_mapping = {
    'Daily': 30,     # 30 times per month
    'Weekly': 4,     # 4 times per month  
    'Monthly': 1     # 1 time per month
}

# Create a new numeric column
df['app_usage_frequency_numeric'] = df['app_usage_frequency'].map(frequency_mapping)

# list of the main KPIs to summarize
numeric_kpis = [
    "total_transactions",         # how many transactions each customer made
    "avg_transaction_value",      # average amount per transaction
    "total_spent",                 # total money spent by each customer
    "active_days",                 # number of days the customer was active
    "app_usage_frequency_numeric", # how often the customer used the app, now quantified
    "last_transaction_days_ago",     # how many days since their last transaction
    "customer_satisfaction_score", # feedback score from the customer
    "ltv"                           # lifetime value of the customer
]

# Show overall statistics (count, mean, std, min, percentiles, max) for all KPIs

print("=== Overall KPI Statistics ===")
print(df[numeric_kpis].describe())



# Block 2: Groupby Segmentation Based on Income Level: 

# Splitting the data into groups based on income level
# In the first run, the outputs were not being fully displayed for all the variables, and so I edited the code block to manually print them, to avoid truncation. 

print("\n=== KPI Statistics by Income Level ===")

# Group by income level and iterate through groups
for income_level, group_data in df.groupby("income_level"):
    print(f"\n=== Income Level: {income_level} ===")
    print(f"Number of customers: {len(group_data)}")
    print("\nDescriptive Statistics:")
    print(group_data[numeric_kpis].describe())
    print("-" * 80)



#EXPLANATION (See Methodology section for an elaborate explanation)

#Segmenting by income ensures each group has a consistent financial baseline, so normal KPI behavior can be defined more accurately.
#This makes it possible to spot when metrics like total_spent or session_frequency diverge in ways that would be hidden in overall averages.
#As a result, false signals of growth are filtered out, allowing the model to flag real product or strategy issues in the fintech context.



#=========================================
# Step 2: Visualisation of KPI Conflicts
#=========================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Sample data to simulate grouped KPI statistics by income level
data = {
    'KPI': ['Total Transactions', 'Avg. Transaction Value', 'Total Spent', 'Active Days', 'Customer Satisfaction', 'Lifetime Value (LTV)'],
    'Low Income': [490, 10013, 4980000, 182, 5.42, 510053],
    'Middle Income': [518, 9994, 5110000, 186, 5.56, 522912],
    'High Income': [507, 9843, 4910000, 179, 5.46, 502359]
}

# Convert the data into a pandas DataFrame
df = pd.DataFrame(data)

# Melt the DataFrame so it works well with seaborn for grouped bar plots
# This converts the wide format into long format: each row is a (KPI, Income Group, Value) triplet
df_melted = df.melt(id_vars='KPI', var_name='Income Group', value_name='Value')

# Set the visual style
sns.set(style='whitegrid')

# Create a figure and axis
plt.figure(figsize=(12, 6))

# Create a clustered bar chart using seaborn
# X-axis = KPI, hue = income group, Y-axis = values
sns.barplot(data=df_melted, x='KPI', y='Value', hue='Income Group')

# Rotate the x-axis labels for better readability
plt.xticks(rotation=30, ha='right')

# Add chart title and axis labels
plt.title('Comparison of KPIs by Income Group')
plt.xlabel('Key Performance Indicator (KPI)')
plt.ylabel('Value')

# Move the legend outside the chart
plt.legend(title='Income Group', bbox_to_anchor=(1.05, 1), loc='upper left')

# Adjust layout to make room for rotated labels and legend
plt.tight_layout()

# Show the plot
plt.show()

