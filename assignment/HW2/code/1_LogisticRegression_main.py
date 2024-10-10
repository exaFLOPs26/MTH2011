import numpy as np
from Optimizer import SGD
from logistic_regression
import matplotlib.pyplot as plt



np.random.seed(428)

# Load dataset
iris_train = pd.read_csv("./assignment/HW2/data/iris_train.csv")
iris_test = pd.read_csv("./assignment/HW2/data/iris_test.csv")

X_train = iris_train.drop(columns='target').values
y_train = iris_train["target"].values
X_test = iris_test.drop(columns='target').values
y_test = iris_test["target"].values



num_data, num_features = train_x.shape
num_label = int(train_y.max()) + 1

for i in range(1,num_features):
    column = train_x[:,i]

    mean= np.mean(column)
    stad = np.std(column)
    train_x[:,i] = [((elements-mean)/(stad)) for elements in column]



print('# of Training data : %d \n' % num_data)

batch_size = 32
# ========================= EDIT HERE =========================
"""
Choose param to search. (epoch or lr)
Specify values of the parameter to search, and fix the other.

e.g.)
search_param = 'lr'
num_epochs = 50
learning_rate = [0.1, 0.01, 0.05]
"""

search_param = 'lr'

# HYPERPARAMETERS
num_epochs = 1000
learning_rate = [ 0.08]
# =============================================================

train_results = []
test_results = []
search_space = learning_rate if search_param == 'lr' else num_epochs

for i, space in enumerate(search_space):
    # Build model
    model = LogisticRegression(num_features=num_features)
    optim = SGD()

    if search_param == 'lr':
        model.train(train_x, train_y, batch_size, num_epochs, space, optim)
    else:
        model.train(train_x, train_y, batch_size, space, learning_rate, optim)
    
    ################### Evaluate on train data
    # Inference
    pred, prob = model.eval(train_x)

    train_acc = accuracy(pred, train_y)
    print(f'[Search {i+1}] Accuracy on Train Data : {train_acc:.4f}\n')

    train_results.append(train_acc)

    ################### Evaluate on test data
    # Inference
    pred, prob = model.eval(test_x)

    test_acc = accuracy(pred, test_y)
    print(f'[Search {i+1}] Accuracy on Test Data : {test_acc:.4f}\n')

    test_results.append(test_acc)



"""
Draw scatter plot of search results.
- X-axis: search paramter
- Y-axis: Accuracy (Train, Test respectively)

Put title, X-axis name, Y-axis name in your plot.

Resources
------------
Official document: https://matplotlib.org/3.2.1/api/_as_gen/matplotlib.pyplot.scatter.html
"Data Visualization in Python": https://medium.com/python-pandemonium/data-visualization-in-python-scatter-plots-in-matplotlib-da90ac4c99f9
"""

plt.scatter(search_space, train_results, label='train', marker='x', s=150)
plt.scatter(search_space, test_results, label='test', marker='o', s=150)
plt.legend()
plt.title('Search results')
plt.xlabel(search_param)
plt.ylabel('Accuracy')
plt.savefig('LogisticRegression_results.png')

