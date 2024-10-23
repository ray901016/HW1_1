import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# CRISP-DM Step: Business Understanding and Data Understanding
st.title("Simple Linear Regression with User-Controlled Parameters")
st.write("Adjust the parameters for slope (a), intercept (b), noise, and number of data points.")

# User inputs for slope, intercept, noise, and number of data points
slope = st.slider("Select slope (a)", -10.0, 10.0, 1.0)
intercept = st.slider("Select intercept (b)", -10.0, 10.0, 0.0)
noise = st.slider("Select noise level", 0.0, 10.0, 1.0)
num_points = st.slider("Select number of points", 10, 100, 50)

# CRISP-DM Step: Data Preparation
# Generate random data
np.random.seed(42)
X = 2 * np.random.rand(num_points, 1)
y = slope * X + intercept + noise * np.random.randn(num_points, 1)

# CRISP-DM Step: Modeling
# Fit a linear regression model
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

# Plotting the data
plt.figure(figsize=(10, 6))
plt.scatter(X, y, label="Data Points")
plt.plot(X, y_pred, color='red', label=f"Fitted Line: y = {model.coef_[0][0]:.2f}x + {model.intercept_[0]:.2f}")
plt.xlabel("X")
plt.ylabel("y")
plt.title("Linear Regression Model")
plt.legend()
plt.grid(True)

# CRISP-DM Step: Evaluation
# Display the plot
st.pyplot(plt)

# Display model results
st.write(f"Fitted line equation: y = {model.coef_[0][0]:.2f}x + {model.intercept_[0]:.2f}")

# CRISP-DM Step: Deployment
st.write("This is deployed using Streamlit. You can adjust the parameters above to see how the model changes.")
