# 🎓 Student Score Prediction Using Machine Learning

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Linear%20Regression-green)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![Status](https://img.shields.io/badge/Project-Completed-success)

## 📖 Overview

This project uses **Linear Regression**, a supervised machine learning algorithm, to predict student exam scores based on the number of hours studied.

The objective is to understand the relationship between study time and academic performance while implementing an end-to-end machine learning workflow using Python and Scikit-Learn.

---

## 🎯 Problem Statement

Can a student's exam score be predicted based on the number of hours they study?

This project answers that question by training a Linear Regression model on historical student performance data.

---

## 🛠️ Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Scikit-Learn
* Jupyter Notebook

---

## 📂 Dataset Information

The dataset contains two columns:

| Column | Description             |
| ------ | ----------------------- |
| Hours  | Number of hours studied |
| Scores | Student exam score      |

### Sample Data

| Hours | Scores |
| ----- | ------ |
| 2.5   | 21     |
| 5.1   | 47     |
| 8.5   | 75     |
| 3.5   | 30     |
| 9.2   | 88     |

---

## 🚀 Project Workflow

### 1️⃣ Import Required Libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
```

### 2️⃣ Load Dataset

```python
df = pd.read_csv("score.csv")
```

### 3️⃣ Data Exploration

```python
df.head()
df.info()
df.describe()
```

### 4️⃣ Feature Selection

```python
X = df[['Hours']]
y = df['Scores']
```

### 5️⃣ Split Dataset

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)
```

### 6️⃣ Train Linear Regression Model

```python
model = LinearRegression()

model.fit(X_train, y_train)
```

### 7️⃣ Make Predictions

```python
y_pred = model.predict(X_test)
```

### 8️⃣ Evaluate Model Performance

```python
print("R2 Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
```

---

## 📊 Data Visualization

### Scatter Plot

```python
plt.scatter(X_train, y_train)
plt.xlabel("Hours Studied")
plt.ylabel("Scores")
plt.show()
```

### Regression Line

```python
plt.scatter(X_train, y_train)

plt.plot(
    X_train,
    model.predict(X_train)
)

plt.show()
```

---

## 📈 Machine Learning Model

### Linear Regression Equation

```text
Score = m × Hours + c
```

Where:

* m = Slope (Coefficient)
* c = Intercept

Model Parameters:

```python
print(model.coef_)
print(model.intercept_)
```

---

## 🔮 Future Predictions

Example:

```python
model.predict([[15]])
```

This predicts the score of a student who studies for 15 hours.

---

## 📋 Results

✔ Successfully trained a Linear Regression model

✔ Identified a positive relationship between study hours and scores

✔ Predicted student scores with good accuracy

✔ Visualized the regression line and dataset trends

---

## 📁 Project Structure

```text
StudentScorePrediction/
│
├── StudentScorePrediction.ipynb
├── score.csv
├── README.md
│
└── images/
    ├── scatter_plot.png
    └── regression_line.png
```

---

## 💡 Key Learnings

Through this project, I learned:

* Data Loading using Pandas
* Data Visualization using Matplotlib
* Train-Test Split
* Linear Regression
* Model Evaluation
* Prediction using Machine Learning
* End-to-End ML Workflow

---

## 🔥 Future Enhancements

* Add multiple features for better prediction
* Use Polynomial Regression
* Compare different regression algorithms
* Deploy the model using Flask
* Deploy the model using Streamlit
* Create a user-friendly web application

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

### Run Jupyter Notebook

```bash
jupyter notebook
```

Open:

```text
StudentScorePrediction.ipynb
```

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repository
2. Create a new branch
3. Commit your changes
4. Push the changes
5. Create a Pull Request

---

## 📜 License

This project is licensed under the MIT License.

---

## 👩‍💻 Author

**Chandana**

🎓 Data Science Student
📊 Machine Learning Enthusiast
🐍 Python Developer

If you found this project useful, please consider giving it a ⭐ on GitHub!

