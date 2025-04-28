import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("titanic.csv")

# Display the first few rows of the dataset
print(df.head())

# Display the infromation about the dataset
print(df.info())

# Display the statistical summary of the dataset
print(df.describe())

# Display the correlation matrix

print(df.corr())

# Display the number of missing values in each column
print(df.isnull().sum())


#handle missing values
sns.heatmap(df.isnull(), cbar=False)
plt.title("Missing Values Heatmap")
plt.show()

#Histogram of Age
sns.histplot(df['age'].dropna(), kde=True)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

# Fill missing values in 'age' with the median age
df['age'].fillna(df['age'].median(), inplace=True)

#bar plot of survived
plt.figure(figsize=(8, 6))  
sns.countplot(x='survived', data=df)

# Display the count of survivors and non-survivors
print(df['survived'].value_counts())    

#survival by class
sns.barplot(x='pclass', y='survived', data=df)


#boxplot of age vs survived

sns.boxplot(x='survived', y='age', data=df)


# Display the average age of survivors and non-survivors
print(df.groupby('survived')['age'].mean())

#correlation matrix heatmap
sns.heatmap(df.corr(numeric_only=True), annot=True)

plt.title("Correlation Matrix")
plt.show()

#pairplot of age, fare, pclass and survived
sns.pairplot(df[['age', 'fare', 'pclass', 'survived']], hue='survived')

