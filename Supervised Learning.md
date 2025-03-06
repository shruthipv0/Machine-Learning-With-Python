# Linear Regression with Scikit-Learn

![image](https://github.com/user-attachments/assets/e6abfcec-31fc-4e1f-9560-dd002f2cb23a)

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
![image](https://github.com/user-attachments/assets/20f91144-5e14-4105-a6f1-ec3b3f24c789)


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

![image](https://github.com/user-attachments/assets/f5c5da43-44c2-49dd-b9eb-0d93f4a83579)

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

![image](https://github.com/user-attachments/assets/35eaf889-31fc-4836-b076-c7dad18dfa33)

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

[image](https://github.com/user-attachments/assets/a0756306-ff15-4f8d-ab47-0920b6051bc4)

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

![image](https://github.com/user-attachments/assets/89b39a83-6ae5-47b6-98ea-80af2dca96f9)

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


![image](https://github.com/user-attachments/assets/3a91fda4-0373-4721-8b83-b2e0753accfb)


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

``` python
correlation_values = raw_data.corr()['tip_amount'].drop('tip_amount')
correlation_values.plot(kind='barh', figsize=(10, 6))
```
raw_data.corr(): Creates a matrix showing the correlation between all numerical columns.   
raw_data.corr()['tip_amount']: Extracts just the correlations for 'tip_amount' with all the other columns.  

 
![image](https://github.com/user-attachments/assets/c8ed76ae-eefe-4ee8-831f-890908295a42)

## Dataset Preprocessing
``` python
# extract the labels from the dataframe
y = raw_data[['tip_amount']].values.astype('float32')

# drop the target variable from the feature matrix
proc_data = raw_data.drop(['tip_amount'], axis=1)

# get the feature matrix used for training
X = proc_data.values

# normalize the feature matrix
X = normalize(X, axis=1, norm='l1', copy=False)
```

## Dataset Train/Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```
## Build a Decision Tree Regressor model with Scikit-Learn

criterion: The function used to measure error, we use 'squared_error'.  

max_depth - The maximum depth the tree is allowed to take; we use 8.

```python
# import the Decision Tree Regression Model from scikit-learn
from sklearn.tree import DecisionTreeRegressor

# for reproducible output across multiple function calls, set random_state to a given integer value
dt_reg = DecisionTreeRegressor(criterion = 'squared_error',
                               max_depth=8, 
                               random_state=35)
dt_reg.fit(X_train, y_train)
```

## valuate the Scikit-Learn and Snap ML Decision Tree Regressor Models¶

```python
# run inference using the sklearn model
y_pred = dt_reg.predict(X_test)

# evaluate mean squared error on the test dataset
mse_score = mean_squared_error(y_test, y_pred)
print('MSE score : {0:.3f}'.format(mse_score))

r2_score = dt_reg.score(X_test,y_test)
print('R^2 score : {0:.3f}'.format(r2_score))
```
## Credit Card Fraud Detection with Decision Trees and SVM

```python
# get the set of distinct classes which is [0,1]
labels = raw_data.Class.unique()
# get the count of each class , gives [284315, 492]
sizes = raw_data.Class.value_counts().values

# plot the class value counts
fig, ax = plt.subplots()
ax.pie(sizes, labels=labels, autopct='%1.3f%%')
ax.set_title('Target Variable Value Counts')
plt.show()

```
![image](https://github.com/user-attachments/assets/9eaa3fd4-f8f4-458d-80dc-85b20419bb39)
As shown above, the Class variable has two values: 0 (the credit card transaction is legitimate) and 1 (the credit card transaction is fraudulent). The dataset is highly unbalanced, the target variable classes are not represented equally (more 0s than 1) sorequires special attention. We can bias the model to pay more attention to the samples in the minority class. 

We need to see what feature has an impact on the classes.

``` python
correlation_values = raw_data.corr()['Class'].drop('Class')
correlation_values.plot(kind='barh', figsize=(10, 6))
```
![image](https://github.com/user-attachments/assets/ed8285b5-c87f-4e14-b333-75320e2cca85)

## Dataset Preprocessing

SVM maximises the margin between different classes using a hyperplane.

If the features have vastly different scales (e.g., one feature ranges from 0 to 1 and another ranges from 1000 to 10000), then the feature with a larger range will dominate the calculation of distances. This could cause the model to give too much importance to one feature and overlook others.

Standard Scaling adjusts the features so that they all have the same scale, typically with a mean of 0 and standard deviation of 1. This ensures that features with very large or very small scales don't dominate the distance calculation.
Normalization (e.g., L1 or L2 normalization) rescales the features to have a unit norm (sum of absolute values = 1 or sum of squared values = 1). This ensures that no feature has more impact on the model than another simply because of its scale.

### Example Dataset

| Feature1 | Feature2 |
|----------|----------|
| 1000     | 0.5      |
| 2000     | 1.5      |
| 3000     | 2.5      |

---

### After Standard Scaling:

| Feature1 (Scaled) | Feature2 (Scaled) |
|-------------------|-------------------|
| -1.0              | -1.0              |
| 0.0               | 0.0               |
| 1.0               | 1.0               |

---

### After L1 Normalization:

| Feature1 (Norm) | Feature2 (Norm) |
|-----------------|-----------------|
| 0.9995          | 0.0005          |
| 0.9990          | 0.0010          |
| 0.9990          | 0.0015          |


``` python

#removing the last column which are the "classes" by indexing.
# assigning it by indexing it like this over writes the raw_data instead of calling it by a new variable.
raw_data.iloc[:, 1:30] = StandardScaler().fit_transform(raw_data.iloc[:, 1:30])
data_matrix = raw_data.values

# X: feature matrix (for this analysis, we exclude the Time variable from the dataset)
X = data_matrix[:, 1:30]

# y: labels vector
y = data_matrix[:, 30]

# data normalization
X = normalize(X, norm="l1")

```
## Dataset Train/Test Split

``` python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

```
## Build a Decision Tree Classifier model with Scikit-Learn

```python

w_train = compute_sample_weight('balanced', y_train)

# for reproducible output across multiple function calls, set random_state to a given integer value
dt = DecisionTreeClassifier(max_depth=4, random_state=35)

dt.fit(X_train, y_train, sample_weight=w_train)

```

### Example: Using `compute_sample_weight('balanced')`
Assume we have the following target labels for a dataset:

y_train = [0, 0, 0, 0, 1, 1, 1]  
Class 0 appears 4 times.  
Class 1 appears 3 times.  
Now, calling compute_sample_weight('balanced', y_train) will compute the weights as follows:

Total number of samples = 7  
Number of classes = 2  
Number of samples in class 0 = 4  
Number of samples in class 1 = 3  
The weight for class 0 will be:  

weight for class 0 = 7 / (2 * 4) = 0.875  
The weight for class 1 will be:  

weight for class 1 = 7 / (2 * 3) = 1.1667  
Thus, the weights for each sample will be:  

Samples with label 0 will have a weight of 0.875.  
Samples with label 1 will have a weight of 1.1667.  
Now, the model will train using these sample weights, so it will "pay more attention" to the underrepresented class 1.  

## Build a Support Vector Machine model with Scikit-Learn

``` python

# for reproducible output across multiple function calls, set random_state to a given integer value
svm = LinearSVC(class_weight='balanced', random_state=31, loss="hinge", fit_intercept=False)

svm.fit(X_train, y_train)
```
## Evaluate the Decision Tree Classifier Models

```python
y_pred_dt = dt.predict_proba(X_test)[:,1]

# ROC-AUC - Receiver Operating Characteristic Curve, higher the value better the seperation of classes is
roc_auc_dt = roc_auc_score(y_test, y_pred_dt)
print('Decision Tree ROC-AUC score : {0:.3f}'.format(roc_auc_dt))

```
## Evaluate the Support Vector Machine Models

```python

y_pred_svm = svm.decision_function(X_test)

roc_auc_svm = roc_auc_score(y_test, y_pred_svm)
print("SVM ROC-AUC score: {0:.3f}".format(roc_auc_svm))

```

# K-Nearest Neighbors Classifier

![image](https://github.com/user-attachments/assets/e93719b8-a934-4b9b-ab84-cf9ac2fc6279)

## Telecommunications customer Classification

``` python
# Finding which features have the highest co relation with customer class
correlation_values = abs(df.corr()['custcat'].drop('custcat')).sort_values(ascending=False)
```

Output shows retire and gender have the least effect on custcat while ed and tenure have the most effect.

## Data preprocessing 

``` python

# Getting X and y values from data
X = df.drop('custcat',axis=1)
y = df['custcat']

# Normalising data
X_norm = StandardScaler().fit_transform(X)

#Train test split
X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.2, random_state=4)

#KNN classification
k = 3
#Train Model and Predict  
knn_classifier = KNeighborsClassifier(n_neighbors=k)
knn_model = knn_classifier.fit(X_train,y_train)

#Predict
yhat = knn_model.predict(X_test)

#Accuracy
print("Test set Accuracy: ", accuracy_score(y_test, yhat))

```

## Choosing the correct value of k

We need the correct value of k for accuracy so we train the model for a different sets of k and check the accuracy to see which value of k is best.

``` python

Ks = 10
#Array to store accuracy values for each K
acc = np.zeros((Ks))

#Array to store standard deviation of accuracy
std_acc = np.zeros((Ks))

for n in range(1,Ks+1):
    #Train Model and Predict  
    knn_model_n = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    yhat = knn_model_n.predict(X_test)

    #Accuracy stored in the index n-1
    acc[n-1] = accuracy_score(y_test, yhat)
    std_acc[n-1] = np.std(yhat==y_test)/np.sqrt(yhat.shape[0])

#Plotting the accuracy of each k

plt.plot(range(1,Ks+1),acc,'g')
plt.fill_between(range(1,Ks+1),acc - 1 * std_acc,acc + 1 * std_acc, alpha=0.10)
plt.legend(('Accuracy value', 'Standard Deviation'))
plt.ylabel('Model Accuracy')
plt.xlabel('Number of Neighbors (K)')
plt.tight_layout()
plt.show()

print( "The best accuracy was with", acc.max(), "with k =", acc.argmax()+1)

#Outputs k = 9
```
![image](https://github.com/user-attachments/assets/f980cc7e-c915-4ccb-abbc-3047220185eb)

## Comparing Random Forest and XGBoost modeling performance

### Random Forest

![image](https://github.com/user-attachments/assets/baa384c4-f9d8-423f-b860-13fc2442602c)


- Random forest model is used for both __classification__ and __regression__.
- Creates a forest of __decision trees__ (many independent decision trees).
- For classification, majority vote from all the trees is taken.
- For regression, average values predicted by all the trees is taken.

![image](https://github.com/user-attachments/assets/74d35a4e-0ede-47aa-b51f-b96ccbe3c349)

## XGB Boost (extreme gradient boosting)

- Builds decision tree and learns what mistakes it has made and creates a new one and improved one.
- Builds trees sequentially.
- Builds a tree, calculates the accuracy (test, predict), builds another tree correcting these errors.

### Comparison: Random Forest vs XGBoost

| Feature               | Random Forest 🌲               | XGBoost ⚡                      |
|----------------------|--------------------------------|--------------------------------|
| **Tree Building**    | Parallel (many trees at once)  | Sequential (one tree at a time) |
| **Speed**           | Slower                         | Faster (optimised)             |
| **Accuracy**        | Good                           | Often better                   |
| **Handles Missing Data** | No (needs preprocessing)  | Yes (automatically handles)     |
| **Overfitting**      | Can overfit                   | Less likely to overfit         |

```python
# Initialize models
n_estimators=100
rf = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
xgb = XGBRegressor(n_estimators=n_estimators, random_state=42)

# Fit models
# Measure how long it takes to train Random Forest
start_time_rf = time.time()
rf.fit(X_train, y_train)
end_time_rf = time.time()
rf_train_time = end_time_rf - start_time_rf

# Measure how long it takes to train XGBoost
start_time_xgb = time.time()
xgb.fit(X_train, y_train)
end_time_xgb = time.time()
xgb_train_time = end_time_xgb - start_time_xgb

print(rf_train_time,xgb_train_time)

#Outputs 17.46025276184082 0.29244160652160645 . So Xgb takes shortest time to train
```

