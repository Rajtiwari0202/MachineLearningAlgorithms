class LinearRegression:

    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.m = 0
        self.b = 0

    def fit(self, x, y):

        n = len(x)

        for epoch in range(self.epochs):

            dm = 0
            db = 0

            for i in range(n):

                prediction = self.m * x[i] + self.b

                error = y[i] - prediction

                dm += x[i] * error
                db += error

            dm = -(2 / n) * dm
            db = -(2 / n) * db

            self.m = self.m - self.learning_rate * dm
            self.b = self.b - self.learning_rate * db

    def predict(self, x):
        return self.m * x + self.b


x = [1, 2, 3, 4]
y = [20, 40, 60, 80]

model = LinearRegression()

model.fit(x, y)

print(f"m = {model.m:.4f}")
print(f"b = {model.b:.4f}")

print(f"Prediction for 5 hours = {model.predict(5):.4f}")