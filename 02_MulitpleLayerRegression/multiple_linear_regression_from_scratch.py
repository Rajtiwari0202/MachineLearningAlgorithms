class MultipleLinearRegression:

    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = []
        self.b = 0

    def fit(self, x, y):

        n = len(x)
        number_of_features = len(x[0])

        self.weights = [0 for _ in range(number_of_features)]
        self.b = 0

        for epoch in range(self.epochs):

            dw = [0 for _ in range(number_of_features)]
            db = 0

            for i in range(n):

                prediction = self.b

                for j in range(number_of_features):
                    prediction += self.weights[j] * x[i][j]

                error = y[i] - prediction

                for j in range(number_of_features):
                    dw[j] += x[i][j] * error

                db += error

            for j in range(number_of_features):
                dw[j] = -(2 / n) * dw[j]

            db = -(2 / n) * db

            for j in range(number_of_features):
                self.weights[j] = self.weights[j] - self.learning_rate * dw[j]

            self.b = self.b - self.learning_rate * db

    def predict(self, x):

        prediction = self.b

        for j in range(len(self.weights)):
            prediction += self.weights[j] * x[j]

        return prediction


x = [
    [1, 1],
    [2, 1],
    [3, 2],
    [4, 2],
]

y = [30, 50, 80, 100]

model = MultipleLinearRegression()

model.fit(x, y)

print(f"weights = {[round(weight, 4) for weight in model.weights]}")
print(f"b = {model.b:.4f}")

print(f"Prediction for 5 hours and 3 assignments = {model.predict([5, 3]):.4f}")
