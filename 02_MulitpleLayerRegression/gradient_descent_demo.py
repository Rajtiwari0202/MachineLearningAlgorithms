# Tiny Dataset
# Features: [hours studied, assignments completed]
x = [
    [1, 1],
    [2, 1],
    [3, 2],
    [4, 2],
]

y = [30, 50, 80, 100]

# Initialize parameters
weights = [0, 0]
b = 0

# Hyperparameters
learning_rate = 0.01
epochs = 1000

# Training Loop
for epoch in range(epochs):

    # Gradient accumulators
    dw = [0 for _ in weights]
    db = 0

    # Used to calculate MSE
    total_error = 0

    # Loop through all training examples
    for i in range(len(x)):

        # Prediction
        prediction = b

        for j in range(len(weights)):
            prediction += weights[j] * x[i][j]

        # Error
        error = y[i] - prediction

        # Accumulate squared error for MSE
        total_error += error ** 2

        # Accumulate gradients for every weight
        for j in range(len(weights)):
            dw[j] += x[i][j] * error

        # Accumulate bias gradient
        db += error

    # Number of samples
    n = len(x)

    # Mean Squared Error
    mse = total_error / n

    # Compute gradients
    for j in range(len(weights)):
        dw[j] = -(2 / n) * dw[j]

    db = -(2 / n) * db

    # Gradient Descent Update
    for j in range(len(weights)):
        weights[j] = weights[j] - learning_rate * dw[j]

    b = b - learning_rate * db

    # Print progress every 100 epochs
    if epoch % 100 == 0:
        print(
            f"Epoch {epoch}: "
            f"MSE={mse:.4f}, "
            f"weights={[round(weight, 4) for weight in weights]}, "
            f"b={b:.4f}"
        )

# Final learned parameters
print("\nTraining Complete!")
print(f"Final weights = {[round(weight, 4) for weight in weights]}")
print(f"Final b = {b:.4f}")
