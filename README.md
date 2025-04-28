This project performs Exploratory Data Analysis (EDA) on the Titanic dataset to uncover patterns, relationships, and anomalies. The analysis uses Python libraries like Pandas, Matplotlib, and Seaborn to derive insights about survival rates, passenger demographics, and other trends.

Dataset
Source: Titanic Dataset
Features:

survived: Survival status (0 = No, 1 = Yes)

pclass: Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd)

sex, age, fare, embarked, etc.

Tools Used
Python

Libraries: Pandas, Matplotlib, Seaborn

Jupyter Notebook,VSCode


bash
pip install pandas matplotlib seaborn jupyter
Run the Jupyter Notebook:

bash
jupyter notebook titanic_eda.ipynb
Generate PDF Report:

From Jupyter Notebook: File > Download as > PDF.

EDA Workflow
1. Data Loading
python
import pandas as pd
df = pd.read_csv("titanic.csv")
2. Initial Exploration
Basic Info:

python
df.info()  # Check data types and missing values
df.describe()  # Summary statistics
Key Observations:

Missing values in age (20%), deck (77%).

38% passengers survived.

3. Visualizations
Univariate Analysis
Age Distribution:

python
sns.histplot(df['age'].dropna(), kde=True)
Observation: Most passengers were 20–40 years old.

Bivariate Analysis
Survival by Passenger Class:

python
sns.barplot(x='pclass', y='survived', data=df)
Observation: 1st-class passengers had a 63% survival rate vs. 24% for 3rd-class.

Multivariate Analysis
Correlation Heatmap:

python
sns.heatmap(df.corr(), annot=True)
Observation: fare and pclass are negatively correlated (-0.55).

Key Findings
Survival Drivers:

Gender: 74% females survived vs. 19% males.

Class: Higher survival in 1st class.

Age: Children (<10) had higher survival rates.

Anomalies:

Fare Outliers: A few passengers paid > £500.

Missing Data: age and deck require imputation.
