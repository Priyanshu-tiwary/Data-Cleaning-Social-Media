import pandas as pd
import numpy as np
df = pd.read_csv("sample_data.csv")

print(df.head())

print(df.info())
print(df.isnull().sum())

# Numeric columns fill with mean
df['likes_count'].fillna(df['likes_count'].mean(), inplace=True)
df['shares_count'].fillna(df['shares_count'].mean(), inplace=True)
df['comments_count'].fillna(df['comments_count'].mean(), inplace=True)

# Categorical fill with mode
df['platform'].fillna(df['platform'].mode()[0], inplace=True)
df['language'].fillna(df['language'].mode()[0], inplace=True)

df.drop_duplicates(inplace=True)

Q1 = df['likes_count'].quantile(0.25)
Q3 = df['likes_count'].quantile(0.75)

IQR = Q3 - Q1

df = df[(df['likes_count'] >= Q1 - 1.5 * IQR) & 
        (df['likes_count'] <= Q3 + 1.5 * IQR)]

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

df[['likes_count', 'shares_count', 'comments_count']] = scaler.fit_transform(
    df[['likes_count', 'shares_count', 'comments_count']]
)

df = pd.get_dummies(df, columns=['platform', 'sentiment_label'], drop_first=True)

df['total_engagement'] = df['likes_count'] + df['shares_count'] + df['comments_count']

df.to_csv("cleaned_data.csv", index=False)

print("Data cleaned and saved successfully!")