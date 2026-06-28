# Multiple Linear Regression

Multiple Linear Regression is a supervised machine learning algorithm used to predict continuous values using more than one input feature.

Example:
Hours Studied + Assignments Completed -> Marks Obtained

In simple Linear Regression, the model uses one feature:

y = mx + b

In Multiple Linear Regression, the model uses many features:

y = w1x1 + w2x2 + w3x3 + ... + b

where:
- w1, w2, w3 = weights for each feature
- x1, x2, x3 = input feature values
- b = bias or intercept

Each weight tells us how important its feature is for the final prediction.

For one training example:

prediction = sum(weight * feature) + bias

The difference between the actual value and predicted value is called error:

error = actual - prediction

To measure the total error across all examples, we use Mean Squared Error (MSE):

MSE = (1/n) * sum(actual - prediction)^2

The objective is to find weights and bias that minimize this error.

To minimize the cost, we use Gradient Descent:

parameter = parameter - learning_rate * gradient

For every epoch, the model:

- predicts the output for every training example
- calculates the error
- accumulates gradients for each weight
- accumulates the gradient for the bias
- updates all weights and bias

Multiple Linear Regression is useful when one feature is not enough to explain the target value.
