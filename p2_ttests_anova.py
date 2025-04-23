#======================================================
# TESTING STATISTICAL SIGNIFICANCE: T-TESTS AND ANOVA
#======================================================

#-----------------
# T-TEST
#-----------------


# Import libraries
import pandas as pd
from scipy.stats import ttest_ind

# Load your cleaned dataset
df = pd.read_csv("fintech_itv_cleaned.csv")

# Drop rows with missing LTV or satisfaction values
df = df.dropna(subset=['ltv', 'customer_satisfaction_score'])

# Step 1: Define threshold – split users into high vs low LTV based on median
ltv_median = df['ltv'].median()

# Step 2: Create two groups: High LTV and Low LTV
high_ltv_group = df[df['ltv'] > ltv_median]
low_ltv_group = df[df['ltv'] <= ltv_median]

# Step 3: Extract satisfaction scores for both groups
satisfaction_high = high_ltv_group['customer_satisfaction_score']
satisfaction_low = low_ltv_group['customer_satisfaction_score']

# Step 4: Run independent t-test (unequal group sizes are okay)
t_stat, p_val = ttest_ind(satisfaction_high, satisfaction_low)

# Step 5: Print results
print(f"T-statistic: {t_stat:.3f}")
print(f"P-value: {p_val:.4f}")

# Step 6: Optional interpretation
if p_val < 0.05:
    print("✅ Statistically significant difference in satisfaction between high and low LTV users.")
else:
    print("❌ No statistically significant difference in satisfaction between high and low LTV users.")



#-----------------
# ANOVA TEST
#-----------------

# Import required libraries

from scipy.stats import f_oneway

# Load the cleaned dataset
df = pd.read_csv("fintech_itv_cleaned.csv")

# Drop any rows with missing income or LTV values
df = df.dropna(subset=['income_level', 'ltv'])

# Step 1: Create three groups based on income level
ltv_low_income = df[df['income_level'] == 'Low']['ltv']
ltv_middle_income = df[df['income_level'] == 'Middle']['ltv']
ltv_high_income = df[df['income_level'] == 'High']['ltv']

# Step 2: Run one-way ANOVA to compare means across the three groups
f_stat, p_val = f_oneway(ltv_low_income, ltv_middle_income, ltv_high_income)

# Step 3: Print the results
print(f"F-statistic: {f_stat:.3f}")
print(f"P-value: {p_val:.4f}")

# Step 4: Optional interpretation
if p_val < 0.05:
    print("✅ Statistically significant differences in LTV across income groups.")
else:
    print("❌ No statistically significant differences in LTV across income groups.")
