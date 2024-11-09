import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import resample
from sklearn.model_selection import train_test_split


df = pd.read_csv("loan_data_test.csv")

#print(df.head())

#print(len(df))

###check for duplicates
duplicates = df[df.duplicated()]
#print(duplicates)
df.drop_duplicates(inplace=True)

###check for missing values
#print(df.isna().sum())

##person_income, person_home_ownership, credit_score, previous_loan_defaults_on_file have little missing values so drop them
df.dropna(subset=['person_income', 'person_home_ownership', 'credit_score',
                  'previous_loan_defaults_on_file', 'person_gender'], inplace=True)

##replace missing values in person_education, loan_intent with mode
#print(df['person_education'].unique())
education_mode = df.person_education.mode()[0]
#print("Mode for education: ",education_mode)
df.person_education.fillna(education_mode,inplace=True)

#print(df['loan_intent'].unique())
loan_intent_mode = df.loan_intent.mode()[0]
#print("Mode for loan intent: ",education_mode)
df.loan_intent.fillna(loan_intent_mode,inplace=True)

##replace missing values in person_age with mean
age_mean = df.person_age.mean()
#print("Mean of price column: ",age_mean)
df.person_age.fillna(age_mean,inplace=True)

##check for other inconsistent values
#print(df['person_gender'].unique())
#print(df['person_home_ownership'].unique())
#print(df['loan_intent'].unique())
#print(df['previous_loan_defaults_on_file'].unique())

#only gender col has "unknown" values, drop them
df = df[df['person_gender'] != 'unknown']

df.info() #check types of cols and if there are any other null values

###convert person_age, credit_score, and cb_person_cred_hist_length to int type for consistency
df['person_age'] = df['person_age'].astype(int)
df['credit_score'] = df['credit_score'].astype(int)
df['cb_person_cred_hist_length'] = df['cb_person_cred_hist_length'].astype(int)

###Check for negative values in each column
# person_income
df = df[df['person_income'] >= 0] #there are negative values here
df.info() #check types of cols and if there are any other null values
#person_age
#negative_values = df[df['person_age'] < 0]
#print((negative_values))

#loan_int_rate here
#negative_values = df[df['loan_int_rate'] < 0]
#print((negative_values))
df = df[df['loan_int_rate'] >= 0]

#credit_score
#negative_values = df[df['credit_score'] < 0]
#print((negative_values))
df = df[df['credit_score'] >= 0]

#no negative values in loan_percent_income
#no negative values in cb_person_cred_hist_length
#no negative values in person_emp_exp
#no negative values in loan_amount
#no negative values in loan_status


# Save the cleaned data to a new CSV file
df.to_csv('cleaned_data.csv', index=False)