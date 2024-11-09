import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Load data
df = pd.read_csv("cleaned_data.csv")

# Replacing Outliers with Median
median_age = df['person_age'].median()
df['person_age'] = df['person_age'].apply(lambda x: median_age if x > 100 else x)

# Encode categorical variables
df['previous_loan_defaults_on_file'] = df['previous_loan_defaults_on_file'].map({'Yes': 1, 'No': 0})
df['person_gender'] = df['person_gender'].map({'male': 1, 'female': 0})

# Label encoding for person_education
label_encoder_person_education = LabelEncoder()
df['person_education'] = label_encoder_person_education.fit_transform(df['person_education'])

# One-hot encoding for categorical variables
df = pd.get_dummies(df, columns=['loan_intent', 'person_home_ownership'])

# Check the class distribution
class_distribution = df['loan_status'].value_counts()
print(class_distribution)

# Plot the class distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='loan_status', data=df, palette=['skyblue', 'lightcoral'])
plt.title('Class Distribution')
plt.xlabel('Class')
plt.ylabel('Number of Instances')
plt.xticks(ticks=[0, 1], labels=['Rejected', 'Approved'], rotation=0)
plt.show()

# Check gender distribution
gender_distribution = df['person_gender'].value_counts()
print(gender_distribution)

plt.figure(figsize=(6, 4))
sns.countplot(x='person_gender', data=df, palette=['skyblue', 'lightcoral'])
plt.title('Gender Distribution')
plt.xlabel('Gender')
plt.ylabel('Number of Instances')
plt.xticks(ticks=[0, 1], labels=['Female', 'Male'], rotation=0)
plt.show()

# Check previous_loan_defaults_on_file distribution
gender_distribution = df['previous_loan_defaults_on_file'].value_counts()
print(gender_distribution)

plt.figure(figsize=(6, 4))
sns.countplot(x='previous_loan_defaults_on_file', data=df, palette=['skyblue', 'lightcoral'])
plt.title('previous_loan_defaults_on_file Distribution')
plt.xlabel('previous_loan_defaults_on_file')
plt.ylabel('Number of Instances')
plt.xticks(ticks=[0, 1], labels=['No', 'Yes'], rotation=0)
plt.show()

# Find correlation
corr = df.corr(method='pearson')
# Plot correlation matrix with clustering and adjusted font sizes for better readability
sns.clustermap(corr, annot=True, fmt=".2f", linewidths=.5, cmap="coolwarm", annot_kws={"size": 5}, figsize=(12, 10))
plt.title("Correlation Matrix Heatmap (Clustered)")
plt.show()