import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
import seaborn as sns
import xgboost as xgb
from xgboost import XGBClassifier

### LOGISTIC REGRESSION  && RANDOM FOREST MODEL

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

# Separate features (X) and target (y)
X = df.drop(columns=['loan_status'])  # Assuming 'loan_status' is the target column
y = df['loan_status']

# Identify numerical columns for scaling
numerical_features = ['person_age', 'person_income', 'person_emp_exp', 'loan_amnt',
                      'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length', 'credit_score']

# Initialize the StandardScaler and scale numerical features
scaler = StandardScaler()
X[numerical_features] = scaler.fit_transform(X[numerical_features])

# Split the data into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Combine X_train and y_train for undersampling
train_data = pd.concat([X_train, y_train], axis=1)

# Separate the majority and minority classes
majority_class = train_data[train_data['loan_status'] == 0]
minority_class = train_data[train_data['loan_status'] == 1]

# Downsample the majority class
majority_class_downsampled = resample(majority_class,
                                      replace=False,
                                      n_samples=len(minority_class),
                                      random_state=42)
# Combine the downsampled majority class with the minority class
balanced_train_data = pd.concat([majority_class_downsampled, minority_class])

# Separate features and target again after balancing
X_train_balanced = balanced_train_data.drop(columns=['loan_status'])
y_train_balanced = balanced_train_data['loan_status']

# Initialize and train the logistic regression model
logistic_model = LogisticRegression()
logistic_model.fit(X_train_balanced, y_train_balanced)

# Predict on the test set using logistic regression
y_pred_logistic = logistic_model.predict(X_test)

# Evaluate the logistic regression model
accuracy_logistic = accuracy_score(y_test, y_pred_logistic)
conf_matrix_logistic = confusion_matrix(y_test, y_pred_logistic)
class_report_logistic = classification_report(y_test, y_pred_logistic)

# Initialize and train the random forest model
random_forest_model = RandomForestClassifier()
random_forest_model.fit(X_train_balanced, y_train_balanced)



# Get feature importances
importances = random_forest_model.feature_importances_
feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

print(feature_importances)

# Plot feature importances
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importances)
plt.title('Feature Importances from Random Forest')
plt.show()



# Predict on the test set using random forest
y_pred_rf = random_forest_model.predict(X_test)

# Evaluate the random forest model
accuracy_rf = accuracy_score(y_test, y_pred_rf)
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
class_report_rf = classification_report(y_test, y_pred_rf)

# Visualize confusion matrix for logistic regression
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
sns.heatmap(conf_matrix_logistic, annot=True, fmt='g', cmap='Blues',
            xticklabels=['Approved', 'Rejected'], yticklabels=['Approved', 'Rejected'])
plt.title('Logistic Regression Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# Visualize confusion matrix for random forest
plt.subplot(1, 2, 2)
sns.heatmap(conf_matrix_rf, annot=True, fmt='g', cmap='Blues',
            xticklabels=['Approved', 'Rejected'], yticklabels=['Approved', 'Rejected'])
plt.title('Random Forest Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.tight_layout()
plt.show()

# Print the evaluation metrics for logistic regression
print("Logistic Regression Accuracy:", accuracy_logistic)
print("\nLogistic Regression Confusion Matrix:\n", conf_matrix_logistic)
print("\nLogistic Regression Classification Report:\n", class_report_logistic)

# Print the evaluation metrics for random forest
print("Random Forest Accuracy:", accuracy_rf)
print("\nRandom Forest Confusion Matrix:\n", conf_matrix_rf)
print("\nRandom Forest Classification Report:\n", class_report_rf)
