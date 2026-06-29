import math


class LogisticRegression:

    def __init__(self, learning_rate=0.1, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = []
        self.b = 0

    def sigmoid(self, z):
        return 1 / (1 + math.exp(-z))

    def fit(self, x, y):

        n = len(x)
        number_of_features = len(x[0])

        self.weights = [0 for _ in range(number_of_features)]
        self.b = 0

        for epoch in range(self.epochs):

            dw = [0 for _ in range(number_of_features)]
            db = 0

            for i in range(n):

                z = self.b

                for j in range(number_of_features):
                    z += self.weights[j] * x[i][j]

                prediction = self.sigmoid(z)
                error = prediction - y[i]

                for j in range(number_of_features):
                    dw[j] += x[i][j] * error

                db += error

            for j in range(number_of_features):
                dw[j] = dw[j] / n

            db = db / n

            for j in range(number_of_features):
                self.weights[j] = self.weights[j] - self.learning_rate * dw[j]

            self.b = self.b - self.learning_rate * db

    def predict_probability(self, x):

        z = self.b

        for j in range(len(self.weights)):
            z += self.weights[j] * x[j]

        return self.sigmoid(z)

    def predict(self, x):

        probability = self.predict_probability(x)

        if probability >= 0.5:
            return 1

        return 0


x = [
    [1, 0.40],
    [2, 0.50],
    [3, 0.60],
    [4, 0.70],
    [5, 0.80],
]

y = [0, 0, 1, 1, 1]

model = LogisticRegression()

model.fit(x, y)

print(f"weights = {[round(weight, 4) for weight in model.weights]}")
print(f"b = {model.b:.4f}")

test_data = [3.5, 0.65]

print(f"Prediction probability = {model.predict_probability(test_data):.4f}")
print(f"Predicted class = {model.predict(test_data)}")
