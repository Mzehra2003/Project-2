#==================================
# CORRELATION ANALYSIS BY BEHAVIOR
#==================================

import pandas as pd

# Step 1: Load your cleaned dataset
df = pd.read_csv("fintech_itv_cleaned.csv")

# Create a mapping from categorical to numeric
frequency_mapping = {
    'Daily': 30,     # 30 times per month
    'Weekly': 4,     # 4 times per month  
    'Monthly': 1     # 1 time per month
}

# Create a new numeric column
df['app_usage_frequency_numeric'] = df['app_usage_frequency'].map(frequency_mapping)


# Step 2: Choose the metric to define engagement
# We'll use 'app_usage_frequency' for this segmentation
# Calculate the median to separate high and low engagement
median_engagement = df['active_days'].median()

# Step 3: Segment the data into high and low engagement groups
high_engagement = df[df['active_days'] > median_engagement]
low_engagement = df[df['active_days'] <= median_engagement]

# Step 4: Run correlation analysis separately for both segments
high_corr = high_engagement.corr(numeric_only=True)
low_corr = low_engagement.corr(numeric_only=True)

# Step 5: Extract correlation values for key KPI pairs to compare
# Define key KPIs to track
kpi_columns = [
    'ltv', 'total_spent', 'avg_transaction_value', 'total_transactions',
    'customer_satisfaction_score', 'last_transaction_days_ago',
    'app_usage_frequency_numeric', 'active_days'
]

# Filter the correlation matrices to show only selected KPI relationships
high_corr_filtered = high_corr.loc[kpi_columns, kpi_columns]
low_corr_filtered = low_corr.loc[kpi_columns, kpi_columns]

print("=== High Engagement Correlation Matrix ===")
print(high_corr_filtered)

print("\n=== Low Engagement Correlation Matrix ===")
print(low_corr_filtered)
