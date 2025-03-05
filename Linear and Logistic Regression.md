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

# One-hot encoding

Convert categorical variables into numerical format using one-hot encoding.

``` python
# Identifying categorical columns
categorical_columns = scaled_data.select_dtypes(include=['object']).columns.tolist()
categorical_columns.remove('NObeyesdad')  # Exclude target column ( obesity level column)

# Applying one-hot encoding
encoder = OneHotEncoder(sparse_output=False, drop='first')
encoded_features = encoder.fit_transform(scaled_data[categorical_columns])
```
If sparse_output=True (the default), the encoded output is returned as a sparse matrix (a matrix with lots of zeros). 
Row, Column  | Value
------------- | -----
(0, 0)        | 1.0
(1, 1)        | 1.0
(2, 2)        | 1.0
(3, 0)        | 1.0

where (i,j) is the index. This matrix only stores non zero values.

If sparse_output=False, the result will be a dense array, which is a regular NumPy array.

[[1.0, 0.0, 0.0],  
 [0.0, 1.0, 0.0],  
 [0.0, 0.0, 1.0]]

``` python
# Converting to a DataFrame
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_columns))

# Combining with the original dataset
prepped_data = pd.concat([scaled_data.drop(columns=categorical_columns), encoded_df], axis=1)

```

## Encode the target variable

we need to encode the target variable to make it suitable for models, especially when the target is categorical (like different classes or categories).
We dont want to use one hot encoding because there is no point in making n different columns for n different category.

First we convert the target variable column into a category type for efficiency.
Then we use .cat which accesses categorical variables and then .codes which converts those variables into an ordinal category i.e 1, 2, 3.

```python

prepped_data['NObeyesdad'] = prepped_data['NObeyesdad'].astype('category').cat.codes

```

## Separate the input and target data

```python
# Preparing final dataset
X = prepped_data.drop('NObeyesdad', axis=1)
y = prepped_data['NObeyesdad']

```
## Model training and evaluation

``` python

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

```
stratify=y - when splitting a dataset into training and testing sets, specifically to ensure that the target variable (y) is proportional in both the training and testing subsets.

The class distribution in both y_train and y_test will be the same as in y:

y had 60% Class 0 and 40% Class 1, so after stratification:
- y_train will have 60% Class 0 and 40% Class 1
- y_test will also have 60% Class 0 and 40% Class 1

## Logistic Regression with One-vs-All

When you have more than two classes (i.e., a multi-class problem), logistic regression by itself is not directly applicable, because it is a binary classifier (it predicts whether an input belongs to one class or not).

The One-vs-All method provides a way to handle multi-class classification using a binary classifier for each class.

Suppose you have k classes (for example, classes: A, B, C).
For each class, we train a separate binary classifier (logistic regression in this case) that tries to distinguish that particular class from all other classes combined.

For class A, the classifier will label:

- Class A as 1 (positive class)
- Class B and Class C as 0 (negative class)
  
For class B, the classifier will label:

- Class B as 1
- Class A and Class C as 0

Imagine you are classifying animals into three categories: Dog, Cat, Rabbit.

For each of the categories, we train a separate binary classifier:

Dog vs. Not Dog: The classifier tries to separate dogs from cats and rabbits.  
Cat vs. Not Cat: The classifier tries to separate cats from dogs and rabbits.  
Rabbit vs. Not Rabbit: The classifier tries to separate rabbits from dogs and cats.

| Classifier       | Confidence Score (Probability)          |
|-----------------|----------------------------------------|
| Dog vs All    | 0.75     |
| Cat vs All   | 0.60  |
| Rabbit vs All   | 0.20 |

The model chooses Dog because it has the **highest confidence score (0.75).

``` python 
# Training logistic regression model using One-vs-All (default)
model_ova = LogisticRegression(multi_class='ovr', max_iter=1000)
model_ova.fit(X_train, y_train)

# Predictions
y_pred_ova = model_ova.predict(X_test)

# Evaluation metrics for OvA
print("One-vs-All (OvA) Strategy")
print(f"Accuracy: {np.round(100*accuracy_score(y_test, y_pred_ova),2)}%")

```

## Logistic Regression with One Vs One

the model trains a binary classifier for every pair of classes. 

If there are k classes, the model trains k × (k - 1) / 2 classifiers.  
Each classifier learns to distinguish only between two classes at a time.

The model trains three binary classifiers:

Apple vs. Banana  
Apple vs. Orange  
Banana vs. Orange  

## Comparing One-vs-One vs. One-vs-All

| Approach         | Number of Classifiers      | How It Works                           | Pros                           | Cons                                   |
|-----------------|---------------------------|----------------------------------------|-------------------------------|----------------------------------------|
| **One-vs-All (OvA)** | k                         | Each class is trained against all others | Fewer models, efficient       | Struggles with class imbalance        |
| **One-vs-One (OvO)** | k × (k - 1) / 2           | Each class is trained against another class | Works well for small datasets | More classifiers = more computation  |


``` python
# Training logistic regression model using One-vs-One
model_ovo = OneVsOneClassifier(LogisticRegression(max_iter=1000))
model_ovo.fit(X_train, y_train)

# Predictions
y_pred_ovo = model_ovo.predict(X_test)

# Evaluation metrics for OvO
print("One-vs-One (OvO) Strategy")
print(f"Accuracy: {np.round(100*accuracy_score(y_test, y_pred_ovo),2)}%")
```
## Decision Trees

Part of your job is to build a model to find out which drug might be appropriate for a future patient with the same illness. The features of this dataset are the Age, Sex, Blood Pressure, and Cholesterol of the patients, and the target is the drug that each patient responded to.During their course of treatment, each patient responded to one of 5 medications, Drug A, Drug B, Drug C, Drug X and Drug Y.

``` python
label_encoder = LabelEncoder()
my_data['Sex'] = label_encoder.fit_transform(my_data['Sex']) 
my_data['BP'] = label_encoder.fit_transform(my_data['BP'])
my_data['Cholesterol'] = label_encoder.fit_transform(my_data['Cholesterol']) 

```
LabelEncoder() is encodes categorical labels into numbers.  
Assigns random numbers so not usable if the aim is for ordinal encoding where order matters.

| Approach       | When to Use |
|---------------|------------|
| `LabelEncoder()` | When working with scikit-learn ML models and the target variable (y). |
| `.cat.codes`  | When preprocessing data in pandas and want to keep category information. |

``` python
custom_map = {'drugA':0,'drugB':1,'drugC':2,'drugX':3,'drugY':4}
my_data['Drug_num'] = my_data['Drug'].map(custom_map)
```
We are using custom mapping here instead of ordinal encoding because the latter assigns numbers based on an inherent order, which may confuse the model into interpreting higher numbers as 'better' or 'greater' than lower ones (e.g., drugB being 1 and drugA being 0). This can introduce unintended relationships. For tasks like correlation analysis, it's best to use custom mapping, as it avoids creating artificial ordinal relationships and ensures meaningful encoding.


```python
my_data.drop('Drug',axis=1).corr()['Drug_num']

my_data.drop('Drug', axis=1): Removing the 'Drug' column.
.corr(): Calculating the correlation matrix of the remaining columns.
```

### Correlation Matrix Example

|                   | Age   | Blood Pressure | Drug_num |
|-------------------|-------|----------------|----------|
| **Age**           | 1.00  | 0.20           | 0.30     |
| **Blood Pressure**| 0.20  | 1.00           | 0.15     |
| **Drug_num**      | 0.30  | 0.15           | 1.00     |

Correlation of 0.30 between Age and Drug_num, meaning there's a weak positive relationship between Age and Drug type.

``` python

# Plot the count plot
category_counts = my_data['Drug'].value_counts()

#.value_counts() counts the number of unique values.

plt.bar(category_counts.index, category_counts.values, color='blue')
plt.xlabel('Drug')
plt.ylabel('Count')
plt.title('Category Distribution')
plt.xticks(rotation=45)  # Rotate labels for better readability if needed
plt.show()

```
![image](https://github.com/user-attachments/assets/9e2c25b9-de65-4d12-b58f-06fa5cd37d3d)


## Modeling

``` python
y = my_data['Drug']
X = my_data.drop(['Drug','Drug_num'], axis=1)

X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=32)

drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)

drugTree.fit(X_trainset,y_trainset)
```
## Evaluation¶

``` python
tree_predictions = drugTree.predict(X_testset)
print("Decision Trees's Accuracy: ", metrics.accuracy_score(y_testset, tree_predictions))
```
## Visualise the tree

``` python

plot_tree(drugTree)
plt.show()

```

![image](https://github.com/user-attachments/assets/0dfcdfc1-d30b-4e16-86bc-54560f86d902)

# Regression Trees

 The dataset includes information about taxi tip and we will use the trained model to predict the amount of tip paid.

 
