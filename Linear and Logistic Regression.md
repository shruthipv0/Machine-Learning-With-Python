# Linear Regression with Scikit-Learn

## Importing Required Libraries
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
```

## Preparing the Dataset
```python
# Selecting relevant features from the DataFrame
cdf = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]

# Extracting input (X) and output (y) variables
X = cdf.ENGINESIZE.to_numpy()
y = cdf.CO2EMISSIONS.to_numpy()
```
Scikit-learn requires NumPy arrays as input for model training.

## Splitting Data into Training and Testing Sets
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
- `test_size=0.2` → 20% of the data is used for testing.
- `random_state=42` → Ensures reproducibility.

## Training the Linear Regression Model
```python
regressor = linear_model.LinearRegression()

# Reshape X_train to 2D because Scikit-learn expects 2D arrays
regressor.fit(X_train.reshape(-1, 1), y_train)
```
Since there is only one feature (`ENGINESIZE`), `X_train` is a 1D array.
Scikit-learn requires a 2D array of shape `(n,1)`, so we reshape it.

## Visualizing the Model
```python
plt.scatter(X_train, y_train, color='blue')  # Actual data points
plt.plot(X_train, regressor.coef_ * X_train + regressor.intercept_, '-r')  # Regression line
plt.xlabel("Engine Size")
plt.ylabel("CO2 Emissions")
plt.show()
```

## Making Predictions and Evaluating the Model
```python
# Reshape X_test before prediction
y_test_pred = regressor.predict(X_test.reshape(-1, 1))

# Evaluating model performance
print("Mean Absolute Error: %.2f" % mean_absolute_error(y_test, y_test_pred))
print("Mean Squared Error: %.2f" % mean_squared_error(y_test, y_test_pred))
print("Root Mean Squared Error: %.2f" % np.sqrt(mean_squared_error(y_test, y_test_pred)))
print("R2 Score: %.2f" % r2_score(y_test, y_test_pred))
```

# Classification with Logistic Regression

## Predicting Customer Churn

### Data Preprocessing
```python
churn_df = churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip', 'churn']]
X = churn_df.to_numpy()
y = churn_df["churn"].to_numpy()
```

It is also a norm to standardize or normalize the dataset in order to have all the features at the same scale.

```python
from sklearn.preprocessing import StandardScaler
X_norm = StandardScaler().fit(X).transform(X)
```

### Splitting the Dataset
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.2, random_state=4)
```

### Logistic Regression Classifier Modeling
```python
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression().fit(X_train, y_train)
```

Fitting, or in simple terms training, gives us a model that has now learned from the training data and can be used to predict the output variable.

```python
yhat = LR.predict(X_test)
```

A machine learning model like Logistic Regression (LR) gives probabilities.

```python
yhat_prob = LR.predict_proba(X_test)
```

`predict_proba(X_test)` gives the probability of each test data point belonging to class 0 (not churned) and class 1 (churned).

- The first column represents the probability that the customer belongs to class 0 (they will stay).
- The second column represents the probability that the customer belongs to class 1 (they will churn).

#### Example Output
Here’s what the model predicted for the first 10 customers in the test dataset:

| Customer | Probability of Staying (Class 0) | Probability of Churning (Class 1) |
|----------|---------------------------------|---------------------------------|
| 1        | 0.7464                          | 0.2536                          |
| 2        | 0.9267                          | 0.0733                          |
| 3        | 0.8344                          | 0.1656                          |
| 4        | 0.9460                          | 0.0540                          |

### Feature Importance

```python
import pandas as pd
import matplotlib.pyplot as plt
coefficients = pd.Series(LR.coef_[0], index=churn_df.columns[:-1])
coefficients.sort_values().plot(kind='barh')
plt.title("Feature Coefficients in Logistic Regression Churn Model")
plt.xlabel("Coefficient Value")
plt.show()
```

We are plotting the coefficient values to see which features have the most impact. For example, in the equation `y = mx + c`, if `x` is some feature:

- If `m` is -0.2, then it has less impact on churn.
- If `m` is +0.8, then it has a higher impact on churn.

### Performance Evaluation

#### Log Loss

```python
from sklearn.metrics import log_loss
log_loss(y_test, yhat_prob)
```
# Multi-class Classification

## Importing Required Libraries
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsOneClassifier
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings('ignore')

```
The dataset used for this lab is the "Obesity Risk Prediction" dataset, containing 17 attributes and 2,111 samples.

## Example of Variables in the Dataset

| Variable Name                         | Type          | Description                                               |
|---------------------------------------|---------------|-----------------------------------------------------------|
| Gender                                | Categorical   | Gender of the person                                      |
| Age                                   | Continuous    | Age of the person                                         |
| Height                                | Continuous    | Height of the person                                      |
| Weight                                | Continuous    | Weight of the person                                      |
| Family History with Overweight        | Binary        | Does the person have a family member who suffered from overweight? |
| FAVC                                  | Binary        | Does the person eat high-caloric food frequently?         |

## Preprocessing the Data

### Feature Scaling

Because this data might have values that are not only numerical . eg. names or things like that. we want to only select the numerical data to scale. 

### Categorical vs Continuous Variables

Categorical Variables: Variables that take a fixed number of categories or classes. 

- Nominal: No inherent order (e.g., Gender, Colour).
- Ordinal: There is an inherent order (e.g., Low, Medium, High, or 1, 2, 3).

Continuous Variables: Variables that can take any value within a range, such as height, weight, or age (e.g., 25.5, 30.2).

``` python
continuous_columns = data.select_dtypes(include=['float64']).columns.tolist()
```
This will automatically identify the columns with continuous numerical values. For example, the output might look like:
```python
['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
```

## Standard Scaling
``` python
scaler = StandardScaler()
scaled_features = scaler.fit_transform(data[continuous_columns])
```
We're doing this because scaled_feature returns a numpy array. its easier to have it as a data frame with the normalised data along with its columns so we can see what has what value for the features. 

## Converting to a DataFrame
``` python
scaled_df = pd.DataFrame(scaled_features, columns=scaler.get_feature_names_out(continuous_columns))
```

## Combining with the original dataset
```python
scaled_data = pd.concat([data.drop(columns=continuous_columns), scaled_df], axis=1)
```
Now that we have the normalised/scaled data, we want to add this back into our original data with these scaled values. so we take the original data, drop the unscaled columns and concat it with the scaled column. Note axis = 1 means concat it by column side bt side 


