# Tiny Dataset
x = [1, 2, 3, 4]
y = [20, 40, 60, 80]

# Initialize parameters
m = 0
b = 0

# Hyperparameters
learning_rate = 0.01
epochs = 1000

# Training Loop
for epoch in range(epochs):

    # Gradient accumulators
    dm = 0
    db = 0

    # Used to calculate MSE
    total_error = 0

    # Loop through all training examples
    for i in range(len(x)):

        # Prediction
        prediction = m * x[i] + b

        # Error
        error = y[i] - prediction

        # Accumulate squared error for MSE
        total_error += error ** 2

        # Accumulate gradients
        dm += x[i] * error
        db += error

    # Number of samples
    n = len(x)

    # Mean Squared Error
    mse = total_error / n

    # Compute gradients
    dm = -(2 / n) * dm
    db = -(2 / n) * db

    # Gradient Descent Update
    m = m - learning_rate * dm
    b = b - learning_rate * db

    # Print progress every 100 epochs
    if epoch % 100 == 0:
        print(
            f"Epoch {epoch}: "
            f"MSE={mse:.4f}, "
            f"m={m:.4f}, "
            f"b={b:.4f}"
        )

# Final learned parameters
print("\nTraining Complete!")
print(f"Final m = {m:.4f}")
print(f"Final b = {b:.4f}")