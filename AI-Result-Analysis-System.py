# Importing required libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# Step 1: Load the dataset
# The CSV file should contain columns: Name, Sub1, Sub2, Sub3, Sub4, Sub5
data = pd.read_csv('student_marks.csv')

# Step 2: Calculate Total and Average marks
data['Total'] = data.iloc[:, 1:].sum(axis=1)
data['Average'] = data['Total'] / (len(data.columns) - 1)

# Step 3: Assign grades based on average
def grade(avg):
    if avg >= 90:
        return 'O'
    elif avg >= 80:
        return 'A+'
    elif avg >= 70:
        return 'A'
    elif avg >= 60:
        return 'B'
    else:
        return 'C'

data['Grade'] = data['Average'].apply(grade)

# Step 4: Display the analyzed data
print("Student Performance Summary:\n")
print(data)

# Step 5: Visualize the performance using charts
plt.figure(figsize=(8, 5))
plt.bar(data['Name'], data['Average'], color='skyblue', edgecolor='black')
plt.title('Student Performance Analysis')
plt.xlabel('Student Name')
plt.ylabel('Average Marks')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Step 6: Predict future average performance using Linear Regression
X = np.array(range(len(data))).reshape(-1, 1)
y = data['Average']

model = LinearRegression()
model.fit(X, y)

# Predict the next student's possible average
future_index = np.array([[len(data)]])
future_pred = model.predict(future_index)
print("\nPredicted next average performance: {:.2f}".format(future_pred[0]))
