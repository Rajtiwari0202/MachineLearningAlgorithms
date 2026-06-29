# Logistic Regression

Logistic Regression is a supervised machine learning algorithm used for classification problems.

Example:
Hours Studied + Attendance -> Pass or Fail

Linear Regression predicts continuous values.

Logistic Regression predicts probabilities between 0 and 1.

First, the model calculates a linear value:

z = w1x1 + w2x2 + w3x3 + ... + b

where:
- w1, w2, w3 = weights for each feature
- x1, x2, x3 = input feature values
- b = bias or intercept

Then Logistic Regression passes z through the sigmoid function:

sigmoid(z) = 1 / (1 + e^-z)

The sigmoid function converts any number into a value between 0 and 1.

This value can be treated as a probability.

Example:

prediction = 0.82

This means the model predicts an 82 percent chance of class 1.

To convert probability into a class:

- if probability >= 0.5, predict 1
- if probability < 0.5, predict 0

For Logistic Regression, Mean Squared Error is not the best cost function.

Instead, we use Binary Cross-Entropy Loss:

loss = -actual * log(prediction) - (1 - actual) * log(1 - prediction)

The objective is to find weights and bias that minimize this loss.

To minimize the cost, we use Gradient Descent:

parameter = parameter - learning_rate * gradient

For every epoch, the model:

- calculates the linear value z
- applies sigmoid to get a probability
- calculates the error
- accumulates gradients for each weight
- accumulates the gradient for the bias
- updates all weights and bias

Logistic Regression is called regression, but it is mainly used for classification.
