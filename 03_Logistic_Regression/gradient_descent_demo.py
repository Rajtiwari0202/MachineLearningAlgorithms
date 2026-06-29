import math


def sigmoid(z):
    return 1 / (1 + math.exp(-z))


# Tiny Dataset
# Features: [hours studied, attendance percentage scaled between 0 and 1]
x = [
    [1, 0.40],
    [2, 0.50],
    [3, 0.60],
    [4, 0.70],
    [5, 0.80],
]

# Labels: 0 = fail, 1 = pass
y = [0, 0, 1, 1, 1]

# Initialize parameters
weights = [0, 0]
b = 0

# Hyperparameters
learning_rate = 0.1
epochs = 1000

# Training Loop
for epoch in range(epochs):

    # Gradient accumulators
    dw = [0 for _ in weights]
    db = 0

    # Used to calculate binary cross-entropy loss
    total_loss = 0

    # Loop through all training examples
    for i in range(len(x)):

        # Linear function
        z = b

        for j in range(len(weights)):
            z += weights[j] * x[i][j]

        # Probability prediction
        prediction = sigmoid(z)

        # Error
        error = prediction - y[i]

        # Avoid log(0)
        prediction = max(min(prediction, 0.999999), 0.000001)

        # Binary Cross-Entropy Loss
        loss = -y[i] * math.log(prediction) - (1 - y[i]) * math.log(1 - prediction)
        total_loss += loss

        # Accumulate gradients for every weight
        for j in range(len(weights)):
            dw[j] += x[i][j] * error

        # Accumulate bias gradient
        db += error

    # Number of samples
    n = len(x)

    # Average loss
    average_loss = total_loss / n

    # Compute average gradients
    for j in range(len(weights)):
        dw[j] = dw[j] / n

    db = db / n

    # Gradient Descent Update
    for j in range(len(weights)):
        weights[j] = weights[j] - learning_rate * dw[j]

    b = b - learning_rate * db

    # Print progress every 100 epochs
    if epoch % 100 == 0:
        print(
            f"Epoch {epoch}: "
            f"Loss={average_loss:.4f}, "
            f"weights={[round(weight, 4) for weight in weights]}, "
            f"b={b:.4f}"
        )

# Final learned parameters
print("\nTraining Complete!")
print(f"Final weights = {[round(weight, 4) for weight in weights]}")
print(f"Final b = {b:.4f}")

test_data = [3.5, 0.65]
z = b

for j in range(len(weights)):
    z += weights[j] * test_data[j]

probability = sigmoid(z)
predicted_class = 1 if probability >= 0.5 else 0

print(f"Prediction probability = {probability:.4f}")
print(f"Predicted class = {predicted_class}")
