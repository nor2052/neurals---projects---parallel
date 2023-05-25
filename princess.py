import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def forward_pass(x, w1, w2, b1, b2):
    # Calculate the output of the first layer.
    z1 = np.dot(x, w1) + b1
    # Apply the sigmoid function to the output of the first layer.
    a1 = sigmoid(z1)
    # Calculate the output of the second layer.
    z2 = np.dot(a1, w2) + b2
    # Apply the sigmoid function to the output of the second layer.
    a2 = sigmoid(z2)
    # Return the output of the second layer.
    return a2


def backpropagation(x, y, w1, w2, b1, b2):
    # Calculate the error at the output layer.
    error = y - a2
    # Calculate the gradient of the error with respect to the output of the second layer.
    d_a2 = error * sigmoid(a2) * (1 - sigmoid(a2))
    # Calculate the gradient of the error with respect to the weights of the second layer.
    d_w2 = np.dot(d_a2, a1.T)
    # Calculate the gradient of the error with respect to the bias of the second layer.
    d_b2 = d_a2
    # Calculate the error at the first layer.
    error = np.dot(d_a2, w2.T)
    # Calculate the gradient of the error with respect to the output of the first layer.
    d_a1 = error * sigmoid(a1) * (1 - sigmoid(a1))
    # Calculate the gradient of the error with respect to the weights of the first layer.
    d_w1 = np.dot(d_a1, x.T)
    # Calculate the gradient of the error with respect to the bias of the first layer.
    d_b1 = d_a1
    # Update the weights and biases.
    w1 = w1 - d_w1 * learning_rate
    w2 = w2 - d_w2 * learning_rate
    b1 = b1 - d_b1 * learning_rate
    b2 = b2 - d_b2 * learning_rate
    # Return the updated weights and biases.
    return w1, w2, b1, b2


def train(x, y, learning_rate, epochs):
    # Initialize the weights and biases.
    w1 = np.random.randn(2, 4)
    w2 = np.random.randn(4, 1)
    b1 = np.random.randn(4)
    b2 = np.random.randn(1)

    # Train the network for the specified number of epochs.
    for epoch in range(epochs):
        # Forward pass.
        a2 = forward_pass(x, w1, w2, b1, b2)

        # Backpropagation.
        w1, w2, b1, b2 = backpropagation(x, y, w1, w2, b1, b2)

    # Return the trained weights and biases.
    return w1, w2, b1, b2


def predict(x, w1, w2, b1, b2):
    # Forward pass.
    a2 = forward_pass(x, w1, w2, b1, b2)

    # Return the predicted output.
    return a2


# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
# from sklearn.metrics import accuracy_score
# df = pd.read_csv('data.csv')


# # def load_csv(filename):
# # 	dataset = list()
# # 	with open('data.csv', 'r') as file:
# # 		csv_reader = reader(file)
# # 		for row in csv_reader:
# # 			if not row:
# # 				continue
# # 			dataset.append(row)
# # 	return dataset


# y = df.Cured.values
# X = df.drop(["Cured"], axis=1)
# X = (X - np.min(X))/(np.max(X) - np.min(X))
# X = np.round(X, 2)
# x_train, x_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42)
# x_train = np.array(x_train)
# x_test = np.array(x_test)
# y_train = np.array(y_train)[:, np.newaxis]
# y_test = np.array(y_test)[:, np.newaxis]


# def init(input_size, hidden_size, output_size, lr):
#     input_size = input_size
#     hidden_size = hidden_size
#     output_size = output_size
#     lr = lr
#     # init wieghts & biases
#     W1 = np.random.uniform(-2.4/input_size, 2.4 /
#                            input_size, (input_size, hidden_size))
#     b1 = np.zeros((1, hidden_size))
#     W2 = np.random.uniform(-2.4/hidden_size, 2.4 /
#                            hidden_size, (hidden_size, output_size))
#     b2 = np.zeros((1, output_size))


# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))


# def forward(X):
#     # Forward propagation
#     # z1 = X @ W1 + b1
#     result = np.zeros((2338, 13))
#     result2 = np.zeros((2338, 13))
#     for i in range(len(X)):
#         for j in range(len(W1[0])):
#             for k in range(len(W1)):
#                 result[i][j] += X[i][k] * W1[k][j]
#     result += b1
#     outSigmoid1 = sigmoid(result)
#     for i in range(len(outSigmoid1)):
#         for j in range(len(W2[0])):
#             for k in range(len(W2)):
#                 result2[i][j] += outSigmoid1[i][k] * W2[k][j]
#     result2 += b2
#     # z2 = a1@W2 + b2
#     outSigmoid2 = sigmoid(result2)
#     return outSigmoid2


# def backward(X, y, output):
#     output_error = y - output
#     output_delta = output_error * (output * (1 - output))
#     # z1_error = output_delta@W2
#     for i in range(len(output_delta)):
#         for j in range(len(W2[0])):
#             for k in range(len(W2)):
#                 z1_error[i][j] += output_delta[i][k] * W2[k][j]
#     z1_delta = z1_error * (a1 * (1 - a1))
#     # W1 += X.dot(z1_delta) * lr
#     for i in range(len(X)):
#         for j in range(len(z1_delta[0])):
#             for k in range(len(z1_delta)):
#                 W1[i][j] += X[i][k] * z1_delta[k][j]
#     b1 += np.sum(z1_delta, axis=0, keepdims=True) * lr
#     # W2 += a1@output_delta * lr
#     for i in range(len(a1)):
#         for j in range(len(output_delta[0])):
#             for k in range(len(output_delta)):
#                 W2[i][j] += a1[i][k] * output_delta[k][j]
#     W2 *= lr
#     b2 += np.sum(output_delta, axis=0, keepdims=True) * lr


# def train(X, y):
#     output = forward(X)
#     backward(X, y, output)


# def predict(X):
#     return np.round(forward(X))


# index = []
# cost_list2 = []
# b = int(x_train.shape[1])
# init(b, 7, 1, 0.01)
# epochs = 1000
# for i in range(epochs):
#     train(x_train, y_train)
#     # cost = np.mean(np.square(y_train - nn.forward(x_train)))
#     cost = np.mean(np.square(y_train - forward(x_train)))

#     cost = float(np.squeeze(cost))
#     if i % 25 == 0:
#         index.append(i)
#         cost_list2.append(cost)
#         print("Cost after iteration", i, ":", cost)
#     y_pred = nn.predict(x_test)
#     accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy:", accuracy)
# ploat_xy(index, cost_list2)
