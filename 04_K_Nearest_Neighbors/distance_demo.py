import math


def euclidean_distance(point1, point2):
    total = 0

    for i in range(len(point1)):
        total += (point1[i] - point2[i]) ** 2

    return math.sqrt(total)


# Tiny Dataset
# Features: [height in cm, weight in kg]
x_train = [
    [120, 25],
    [130, 30],
    [150, 45],
    [160, 50],
    [175, 70],
    [185, 80],
]

y_train = ["child", "child", "teen", "teen", "adult", "adult"]

# New data point
new_point = [158, 48]

# Number of neighbors
k = 3

distances = []

# Calculate distance from new point to every training point
for i in range(len(x_train)):
    distance = euclidean_distance(new_point, x_train[i])
    distances.append([distance, y_train[i]])

# Sort by smallest distance
distances.sort()

# Select K nearest neighbors
nearest_neighbors = distances[:k]

print("Nearest Neighbors:")

for neighbor in nearest_neighbors:
    print(f"Distance={neighbor[0]:.4f}, Class={neighbor[1]}")

# Majority voting
votes = {}

for neighbor in nearest_neighbors:
    label = neighbor[1]

    if label not in votes:
        votes[label] = 0

    votes[label] += 1

predicted_class = max(votes, key=votes.get)

print(f"\nPredicted class = {predicted_class}")
