import math


class KNearestNeighbors:

    def __init__(self, k=3):
        self.k = k
        self.x_train = []
        self.y_train = []

    def fit(self, x, y):
        self.x_train = x
        self.y_train = y

    def euclidean_distance(self, point1, point2):

        total = 0

        for i in range(len(point1)):
            total += (point1[i] - point2[i]) ** 2

        return math.sqrt(total)

    def predict_one(self, x):

        distances = []

        for i in range(len(self.x_train)):
            distance = self.euclidean_distance(x, self.x_train[i])
            distances.append([distance, self.y_train[i]])

        distances.sort()
        nearest_neighbors = distances[:self.k]

        votes = {}

        for neighbor in nearest_neighbors:
            label = neighbor[1]

            if label not in votes:
                votes[label] = 0

            votes[label] += 1

        return max(votes, key=votes.get)

    def predict(self, x):

        predictions = []

        for sample in x:
            predictions.append(self.predict_one(sample))

        return predictions


x_train = [
    [120, 25],
    [130, 30],
    [150, 45],
    [160, 50],
    [175, 70],
    [185, 80],
]

y_train = ["child", "child", "teen", "teen", "adult", "adult"]

model = KNearestNeighbors(k=3)

model.fit(x_train, y_train)

test_data = [
    [158, 48],
    [180, 75],
]

print(f"Prediction for one person = {model.predict_one([158, 48])}")
print(f"Predictions for multiple people = {model.predict(test_data)}")
