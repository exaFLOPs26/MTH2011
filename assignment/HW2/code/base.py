"""
두개의 붓꽃 종을 로지스틱 회귀로 분류하는 코드를 짜서 업로드 하세요. 
회귀 모형에서 모수를 찾는 방식은 반드시 SGD를 활용해야 합니다. 
파일은 노트북 파일 형식 ipynb로 올리기 바랍니다. 파일은 따로 첨부할게요.
그리고 분류 성능을 잴 수 있도록 테스트셋에 대한 정확도도 함께 명시하기 바랍니다.
마지막으로 두 개의 종에 관한 decision boundary를 표시하는 아래와 같은 그림도 그리기 바랍니다. 그림은 단지 샘플로 올린거니 형식만 참고하세요.

* NumPy, Matplotlib, Pandas 등 기본적 패키지만 사용하고, Scikit-Learn 등 선형 회귀가 구현이 되어져있는 패키지는 응용 패키지는 사용하지 마세요.
2021313075 백경인

score on test set: 

Plan
1. SGD와 minibatch-SGD 비교하기
2. Hyperparameter 다양하게 설정하기

Caution
1. Drop nan data
2. Feature scaling
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
iris_train = pd.read_csv("./assignment/HW2/data/iris_train.csv")
iris_test = pd.read_csv("./assignment/HW2/data/iris_test.csv")

X_train = iris_train.drop(columns='target').values
y_train = iris_train["target"].values
X_test = iris_test.drop(columns='target').values
y_test = iris_test["target"].values

# Add bias term (intercept)
X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Logistic loss function
def loss(X, y, theta):
    m = len(y)
    h = sigmoid(X @ theta.T)
    return -(1/m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))

# SGD for logistic regression
def sgd_logistic_regression(X, y, learning_rate, epochs):
    m, n = X.shape
    theta = np.zeros(n).reshape(1,n)
    
    for epoch in range(epochs):
        for i in range(m):
            rand_idx = np.random.randint(m)
            xi = X[rand_idx:rand_idx+1].T
            yi = y[rand_idx:rand_idx+1]
            gradient = (sigmoid(theta @ xi) - yi) * xi
            theta = theta - learning_rate * gradient.T
        
        # Compute and print the loss every 1000 epochs
        if (epoch + 1) % 1000 == 0:
            current_loss = loss(X, y, theta)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {current_loss:.4f}")
    
    return theta

# Train the model
theta = sgd_logistic_regression(X_train, y_train, learning_rate=0.001, epochs=1000)

# Predict function
def predict(X, theta):
    return np.round(sigmoid(X @ theta.T))

# Accuracy calculation
y_pred = predict(X_test, theta)
accuracy = np.mean(y_pred == y_test)
print(f"Test accuracy: {accuracy * 100:.2f}%")

# Decision boundary plot
def plot_decision_boundary(X, y, theta):
    plt.figure(figsize=(8, 6))

    # Plot the original data points
    plt.scatter(X[:, 1][y == 0], X[:, 2][y == 0], color='blue', label='Class 0')
    plt.scatter(X[:, 1][y == 1], X[:, 2][y == 1], color='red', label='Class 1')

    # Plot the decision boundary
    x_boundary = np.array([min(X[:, 1]) - 1, max(X[:, 1]) + 1])
    y_boundary = -(theta[0] + theta[1] * x_boundary) / theta[2]
    plt.plot(x_boundary, y_boundary, label="Decision Boundary", color='green')

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()

# Plot decision boundary
plot_decision_boundary(X_train, y_train, theta)