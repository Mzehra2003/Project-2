#========================================
# SCATTERPLOTS TO SHOWCASE KPI CONFLICTS
#========================================

# In the first draft of this visualisation, I did a basic heatmap focussed on all KPIs together which showed no conflicts or meaningful insights beyond what had already been established by the comparisons between KPI means (overall as well as Income Level)

# After playing around with different pairings, I settled on the following in order to showcase the best possible scenario
# of misleading KPI signals on dashboards.
# The logic behind each pairing is as follows:
# 1) app usage frequency vs ltv: directly tests if more usage equates to more lifetime value
# 2) customer satisfaction score vs total spent: tests if happier users automatically spend more
# 3) Newer users should be more valuable, if not, thats a churn risk blind spot


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the cleaned dataset 
df = pd.read_csv("fintech_itv_cleaned.csv")

# Convert app usage frequency from category to numeric
frequency_mapping = {'Daily': 30, 'Weekly': 4, 'Monthly': 1}
df['app_usage_frequency_numeric'] = df['app_usage_frequency'].map(frequency_mapping)

# Set style
sns.set(style='whitegrid')

# Create 1x3 subplot grid
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# Plot 1: Usage Frequency vs LTV
sns.regplot(
    ax=axes[0],
    x='app_usage_frequency_numeric',
    y='ltv',
    data=df,
    scatter_kws={'alpha': 0.4},
    line_kws={'color': 'red'}
)
axes[0].set_title('App Usage Frequency vs LTV')
axes[0].set_xlabel('App Usage Frequency (Numeric)')
axes[0].set_ylabel('Lifetime Value (LTV)')

# Plot 2: Satisfaction vs Total Spent
sns.regplot(
    ax=axes[1],
    x='customer_satisfaction_score',
    y='total_spent',
    data=df,
    scatter_kws={'alpha': 0.4},
    line_kws={'color': 'red'}
)
axes[1].set_title('Customer Satisfaction vs Total Spent')
axes[1].set_xlabel('Customer Satisfaction Score')
axes[1].set_ylabel('Total Spent')

# Plot 3: Last Transaction Recency vs LTV
sns.regplot(
    ax=axes[2],
    x='last_transaction_days_ago',
    y='ltv',
    data=df,
    scatter_kws={'alpha': 0.4},
    line_kws={'color': 'red'}
)
axes[2].set_title('Recency vs Lifetime Value')
axes[2].set_xlabel('Days Since Last Transaction')
axes[2].set_ylabel('Lifetime Value (LTV)')

# Layout
plt.tight_layout()
plt.show()

 
