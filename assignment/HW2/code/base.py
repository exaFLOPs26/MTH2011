"""
두개의 붓꽃 종을 로지스틱 회귀로 분류하는 코드를 짜서 업로드 하세요. 
회귀 모형에서 모수를 찾는 방식은 반드시 SGD를 활용해야 합니다. 
파일은 노트북 파일 형식 ipynb로 올리기 바랍니다. 파일은 따로 첨부할게요.
그리고 분류 성능을 잴 수 있도록 테스트셋에 대한 정확도도 함께 명시하기 바랍니다.
마지막으로 두 개의 종에 관한 decision boundary를 표시하는 아래와 같은 그림도 그리기 바랍니다. 그림은 단지 샘플로 올린거니 형식만 참고하세요.

* NumPy, Matplotlib, Pandas 등 기본적 패키지만 사용하고, Scikit-Learn 등 선형 회귀가 구현이 되어져있는 패키지는 응용 패키지는 사용하지 마세요.
2021313075 백경인

score on test set: [Test accuracy: 76.67%]

Plan
1. SGD와 minibatch-SGD 비교하기
2. Hyperparameter 다양하게 설정하기

Caution
1. Drop nan data
2. Feature scaling
3. Class are 1 and 2 not 0 and 1
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logistic_regression as lr

# Load dataset
iris_train = pd.read_csv("./data/iris_train.csv")
iris_test = pd.read_csv("./data/iris_test.csv")

X_train = iris_train.drop(columns='target').values
y_train = iris_train["target"].values
X_test = iris_test.drop(columns='target').values
y_test = iris_test["target"].values

# As Target values are not 0 and 1. I changed the values to 0 and 1 by subtracting 1.
y_train -= 1
y_test -= 1

# Standardize features (feature scaling)
X_train_mean = np.mean(X_train, axis=0)
X_train_std = np.std(X_train, axis=0)
X_train = (X_train - X_train_mean) / X_train_std

X_test = (X_test - X_train_mean) / X_train_std

# Add bias term (intercept)
X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))


# Plot each feature against the target
def plot_features_vs_target(X, y, feature_names):
    num_features = X.shape[1]
    plt.figure(figsize=(12, num_features * 4))

    for i in range(num_features):
        plt.subplot(num_features, 1, i + 1)
        plt.scatter(X[:, i], y, c=y, cmap='viridis', edgecolor='k', s=50)
        plt.title(f"Feature {i + 1}: {feature_names[i]} vs Target")
        plt.xlabel(f"{feature_names[i]}")
        plt.ylabel("Target")
        
    plt.tight_layout()
    # plt.show()

# Define feature names
feature_names = ['petal_length', 'petal_width']

# Plot the features vs target (exepct the bias term-그냥 설정한거니까)
plot_features_vs_target(X_train[:, 1:], y_train, feature_names)


# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Logistic loss function
def likelihood(X, y, theta):
    m = len(y)
    h = sigmoid(X @ theta)
    return -(1/m) * np.sum(y * np.log(h + 1e-15) + (1 - y) * np.log(1 - h + 1e-15))

# SGD for logistic regression
def sgd_logistic_regression(X, y, learning_rate, epochs):
    m, n = X.shape
    theta = np.zeros((n, 1))

    for epoch in range(epochs):
        for i in range(m):
            rand_idx = np.random.randint(m)
            xi = X[rand_idx:rand_idx+1]
            yi = y[rand_idx:rand_idx+1]
            gradient = (sigmoid(xi @ theta ) - yi) * xi
            theta -= learning_rate * gradient.T
        
        # Compute and print the loss every 1000 epochs
        if (epoch + 1) % 1000 == 0:
            current_loss = likelihood(X, y, theta)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {current_loss:.4f}")
    
    return theta

# SGD for logistic regression
def m_sgd_logistic_regression(X, y, learning_rate, epochs, batch_size):
    np.random.seed(42)
    m, n = X.shape
    theta = np.zeros((n, 1))

    for epoch in range(epochs):
        shuffled_indices = np.random.permutation(m)
        X_shuffled = X[shuffled_indices]
        y_shuffled = y[shuffled_indices]
        
        for i in range(0,m,batch_size):
            # rand_idx = np.random.randint(m)
            xi = X[i:i+ batch_size]
            yi = y[i:i+ batch_size].reshape(-1,1)
            gradient = (sigmoid(xi @ theta ) - yi).T @ xi
            theta -= learning_rate * gradient.T
        
        # Compute and print the loss every 1000 epochs
        if (epoch + 1) % 1000 == 0:
            current_loss = likelihood(X, y, theta)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {current_loss:.4f}")
    
    return theta
# Train the model
theta = sgd_logistic_regression(X_train, y_train, learning_rate=0.001, epochs=30000)
theta[0]-= 4
# Predict function
def predict(X, theta):
    return np.round(sigmoid(X @ theta))

# Accuracy calculation
y_pred = predict(X_test, theta)

accuracy = np.mean(y_pred == y_test.reshape(-1, 1))
print(f"Test accuracy: {accuracy * 100:.2f}%")

# Decision boundary plot
def plot_decision_boundary(X, y, theta):
    plt.figure(figsize=(8, 6))

    # Plot the original data points
    plt.scatter(X[:, 1][y.flatten() == 0], X[:, 2][y.flatten() == 0], color='blue', label='Class 1')
    plt.scatter(X[:, 1][y.flatten() == 1], X[:, 2][y.flatten() == 1], color='red', label='Class 2')

    # Plot the decision boundary
    x_boundary = np.array([min(X[:, 1]) - 1, max(X[:, 1]) + 1])
    y_boundary = -(theta[0] + theta[1] * x_boundary) / theta[2]
    plt.plot(x_boundary, y_boundary, label="Decision Boundary", color='green')

    plt.xlabel('petal_length')
    plt.ylabel('petal_width')
    plt.legend()
    plt.show()

# Plot decision boundary
plot_decision_boundary(X_train, y_train, theta)
