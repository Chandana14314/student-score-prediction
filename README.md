# 🎓 Student Score Prediction Using Machine Learning

## 📖 Overview

This project uses **Linear Regression**, a supervised Machine Learning algorithm, to predict student exam scores based on the number of hours studied.

The project demonstrates the complete machine learning workflow, from data loading and visualization to model training and prediction.

---

## 🎯 Problem Statement

Can a student's exam score be predicted based on the number of hours they study?

To answer this question, a Linear Regression model is trained using historical student performance data.

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-Learn
* Jupyter Notebook

---

## 📂 Dataset Information

The dataset contains two columns:

| Column | Description           |
| ------ | --------------------- |
| Hours  | Number of study hours |
| Scores | Student exam score    |

---

## 📁 Project Structure

```text
StudentScorePrediction/
│
├── StudentScorePrediction.ipynb
├── score.csv
└── README.md
```

---

# 🚀 Project Workflow

## Step 1: Import Required Libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
```

### Explanation

* **NumPy** is used for numerical computations.
* **Pandas** is used for data manipulation and analysis.
* **Matplotlib** is used for data visualization.
* **train_test_split** splits data into training and testing sets.
* **LinearRegression** creates the machine learning model.
* **r2_score** and **mean_squared_error** evaluate model performance.

---

## Step 2: Load the Dataset

```python
df = pd.read_csv("score.csv")
```

### Explanation

The dataset is loaded into a Pandas DataFrame for analysis and processing.

Display first few rows:

```python
df.head()
```

---

## Step 3: Explore the Dataset

```python
df.info()
df.describe()
```

### Explanation

These functions help us understand:

* Number of rows and columns
* Data types
* Missing values
* Statistical summary of data

This step ensures the dataset is clean before training the model.

---

## Step 4: Visualize the Data

```python
plt.scatter(df['Hours'], df['Scores'])
plt.xlabel("Hours Studied")
plt.ylabel("Scores")
plt.title("Hours vs Scores")
plt.show()
```

### Explanation

A scatter plot is created to visualize the relationship between study hours and exam scores.

### Observation

The plot shows a positive relationship:

> Students who study more hours generally achieve higher scores.

---

## Step 5: Select Features and Target Variable

```python
X = df[['Hours']]
y = df['Scores']
```

### Explanation

Machine Learning models require:

* **Features (X)** → Input data
* **Target (y)** → Output to predict

Here:

* Hours = Input Feature
* Scores = Target Variable

---

## Step 6: Split the Dataset

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)
```

### Explanation

The dataset is divided into:

* 70% Training Data
* 30% Testing Data

Training data is used to teach the model.

Testing data is used to evaluate performance on unseen data.

---

## Step 7: Train the Model

```python
model = LinearRegression()

model.fit(X_train, y_train)
```

### Explanation

A Linear Regression model is created and trained using the training dataset.

The model learns the relationship between:

```text
Hours Studied → Exam Score
```

---

## Step 8: Make Predictions

```python
y_pred = model.predict(X_test)
```

### Explanation

The trained model predicts scores for the testing dataset.

These predictions are later compared with actual values.

---

## Step 9: Evaluate the Model

```python
print("R2 Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
```

### Explanation

### R² Score

Measures how well the model explains the data.

* Closer to 1 = Better Model
* Closer to 0 = Poor Model

### Mean Squared Error (MSE)

Measures prediction error.

Lower values indicate better performance.

---

## Step 10: Visualize the Regression Line

```python
plt.scatter(X_train, y_train)

plt.plot(
    X_train,
    model.predict(X_train)
)

plt.xlabel("Hours Studied")
plt.ylabel("Scores")

plt.show()
```

### Explanation

The regression line represents the best-fit line learned by the model.

It shows how scores are expected to change as study hours increase.

---

## 📈 Linear Regression Equation

The model follows:

```text
Score = m × Hours + c
```

Where:

* m = Slope (Coefficient)
* c = Intercept

Retrieve values using:

```python
print(model.coef_)
print(model.intercept_)
```

---

## 🔮 Future Prediction Example

```python
model.predict([[15]])
```

### Explanation

This predicts the score of a student who studies for 15 hours.

The model uses the learned relationship to estimate the result.

---

## 📊 Results

✔ Successfully trained a Linear Regression model

✔ Visualized the relationship between study hours and scores

✔ Achieved accurate predictions

✔ Evaluated model performance using standard metrics

---

## 💡 Key Learnings

Through this project, I learned:

* Data Loading using Pandas
* Exploratory Data Analysis (EDA)
* Data Visualization
* Feature Selection
* Train-Test Split
* Linear Regression
* Model Evaluation
* Machine Learning Workflow

---

## 🔥 Future Improvements

* Add more features such as attendance and previous grades
* Use Polynomial Regression
* Compare multiple regression algorithms
* Deploy the model using Streamlit
* Build an interactive web application

---

## ▶️ How to Run the Project

### Clone the Repository

```bash
git clone https://github.com/your-username/StudentScorePrediction.git
```

### Navigate to Project Folder

```bash
cd StudentScorePrediction
```

### Install Required Libraries

```bash
pip install numpy pandas matplotlib scikit-learn
```

### Launch Jupyter Notebook

```bash
jupyter notebook
```

Open:

```text
StudentScorePrediction.ipynb
```

and run all cells.

---

## 👩‍💻 Author

**Chandana**

🎓 Data Science Student

📊 Machine Learning Enthusiast

🐍 Python Developer

---

⭐ If you found this project useful, consider giving it a star on GitHub.


