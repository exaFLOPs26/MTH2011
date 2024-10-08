import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
보스턴 집값을 선형회귀하여 예측하는 코드를 짜서 업로드 하세요.
선형 회귀에서 모수를 찾는 방식은 반드시 SGD를 활용해야 합니다.
파일은 노트북 파일 형식 ipynb로 올리기 바랍니다. 파일은 따로 첨부할게요.

training set과 test set을 7:3으로 나누고 test set에 대해 R^2가 얼마나 나오는지 체크해서 ipynb 파일에 적어두기 바랍니다.
데이터가 506개이니 앞의 350개를 training set, 뒤의 156개를 test set으로 활용하면 될 것 같네요.
--> 과제 수정되어 train, test 주어짐
* NumPy, Matplotlib 등 기본적 패키지만 사용하고, Scikit-Learn 등 선형 회귀가 구현이 되어져있는 패키지는 응용 패키지는 사용하지 마세요.

2021313075 백경인

R^2 score on test set:  0.712776971324815

Plan
1. SGD와 minibatch-SGD 비교하기
2. Hyperparameter 다양하게 설정하기

Caution
1. Drop nan data
2. Feature scaling
"""

# Load the dataset
train_data = pd.read_csv('./assignment/HW1/data/housing_train.csv')
test_data = pd.read_csv('./assignment/HW1/data/housing_test.csv')

# Drop nan data
train_data.dropna(inplace=True)
test_data.dropna(inplace=True)

# Separate features and target variable
X_train = train_data.drop(columns='MEDV').values
y_train = train_data['MEDV'].values
X_test = test_data.drop(columns='MEDV').values
y_test = test_data['MEDV'].values

# Standardize features (feature scaling)
X_train_mean = np.mean(X_train, axis=0)
X_train_std = np.std(X_train, axis=0)
X_train = (X_train - X_train_mean) / X_train_std

X_test = (X_test - X_train_mean) / X_train_std

# Add intercept term (bias) to features
X_train = np.c_[np.ones(X_train.shape[0]), X_train]
X_test = np.c_[np.ones(X_test.shape[0]), X_test]

# Implementing SGD for Linear Regression
def sgd_linear_regression(X, y, learning_rate=0.001, epochs=512):
    np.random.seed(42)
    m, n = X.shape
    theta = np.random.randn(n)  # Initialize parameters randomly
    for epoch in range(epochs):
        for i in range(m):
            random_index = np.random.randint(m)
            xi = X[random_index:random_index+1]
            yi = y[random_index:random_index+1]
            gradients = xi.T.dot(xi.dot(theta) - yi)
            theta = theta - learning_rate * gradients
    return theta

# Implementing Mini-Batch SGD for Linear Regression
def mini_batch_sgd_linear_regression(X, y, learning_rate=0.001, epochs=1000, batch_size=32):
    np.random.seed(42)
    m, n = X.shape
    theta = np.random.randn(n)  # Initialize parameters randomly
    for epoch in range(epochs):
        # Shuffle the dataset at the start of each epoch
        shuffled_indices = np.random.permutation(m)
        X_shuffled = X[shuffled_indices]
        y_shuffled = y[shuffled_indices]
        
        # Iterate over mini-batches
        for i in range(0, m, batch_size):
            xi = X_shuffled[i:i + batch_size]
            yi = y_shuffled[i:i + batch_size]
            
            # Compute the gradient for the batch
            gradients = batch_size * xi.T.dot(xi.dot(theta) - yi)
            
            # Update theta
            theta = theta - learning_rate * gradients
    
    return theta


# Train the model
theta = sgd_linear_regression(X_train, y_train)

# Predict using the model
y_pred = X_test.dot(theta)

# Calculate R^2 score
def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

r2_score = r_squared(y_test, y_pred)

# Print R^2 score
print("R^2 score on test set: ", r2_score)

# Plot predicted vs actual values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', lw=2)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices')
plt.show()