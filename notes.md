# Linear Regression

Linear Regression is a supervised machine learning algorithm used to predict continuous values by modeling a linear relationship between input features and output values.

Example:
Hours Studied -> Marks Obtained

The model tries to fit the best possible line:

y = mx + b

where:
- m = slope
- b = intercept

Real-world data is noisy, so predictions are rarely perfect. The difference between actual and predicted values is called error.

To measure the total error, we use Mean Squared Error (MSE):

MSE = (1/n) * Σ(actual - predicted)^2

Squaring prevents positive and negative errors from cancelling each other and penalizes large mistakes more heavily.

MSE acts as the cost function. The objective of Linear Regression is to find parameters that minimize this cost.

To minimize the cost, we use Gradient Descent:

parameter = parameter - learning_rate * gradient

The gradient tells us the direction of steepest increase of the cost function, so subtracting it moves us toward the minimum error.

Gradient Descent repeatedly updates the parameters until the cost reaches a minimum.