# Linear Regression with Scikit-Learn
```python
1. Importing Required Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

2. Preparing the Dataset

# Selecting relevant features from the DataFrame
cdf = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]

# Extracting input (X) and output (y) variables
X = cdf.ENGINESIZE.to_numpy()
y = cdf.CO2EMISSIONS.to_numpy()
Scikit-learn requires NumPy arrays as input for model training.

3. Splitting Data into Training and Testing Sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
test_size=0.2 → 20% of the data is used for testing.
random_state=42 → Ensures reproducibility.

4. Training the Linear Regression Model

regressor = linear_model.LinearRegression()

# Reshape X_train to 2D because Scikit-learn expects 2D arrays
regressor.fit(X_train.reshape(-1, 1), y_train)
Since there is only one feature (ENGINESIZE), X_train is a 1D array.
Scikit-learn requires a 2D array of shape (n,1), so we reshape it.

5. Visualizing the Model

plt.scatter(X_train, y_train, color='blue')  # Actual data points
plt.plot(X_train, regressor.coef_ * X_train + regressor.intercept_, '-r')  # Regression line
plt.xlabel("Engine Size")
plt.ylabel("CO2 Emissions")
plt.show()

6. Making Predictions and Evaluating the Model

# Reshape X_test before prediction
y_test_pred = regressor.predict(X_test.reshape(-1, 1))

# Evaluating model performance
print("Mean Absolute Error: %.2f" % mean_absolute_error(y_test, y_test_pred))
print("Mean Squared Error: %.2f" % mean_squared_error(y_test, y_test_pred))
print("Root Mean Squared Error: %.2f" % np.sqrt(mean_squared_error(y_test, y_test_pred)))
print("R2 Score: %.2f" % r2_score(y_test, y_test_pred))
