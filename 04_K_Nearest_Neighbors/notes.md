# K-Nearest Neighbors

K-Nearest Neighbors (KNN) is a supervised machine learning algorithm used for classification and regression.

Example:
Height + Weight -> Person Type

KNN is called a lazy learning algorithm because it does not learn weights during training.

Instead, it stores the training data.

When a new data point is given, KNN:

- calculates the distance from the new point to every training point
- sorts the distances from smallest to largest
- selects the nearest K points
- uses majority voting to predict the class

The most common distance formula is Euclidean Distance:

distance = sqrt((x2 - x1)^2 + (y2 - y1)^2)

For more than two features:

distance = sqrt((a1 - b1)^2 + (a2 - b2)^2 + ... + (an - bn)^2)

The value of K controls how many neighbors vote.

If K = 3, the algorithm checks the 3 nearest training examples.

Example:

- neighbor 1 = child
- neighbor 2 = teen
- neighbor 3 = teen

The predicted class is teen because teen gets the majority vote.

Small K values can make the model sensitive to noise.

Large K values can make the model too general.

KNN is simple and powerful, but it can become slow on large datasets because it calculates distances during prediction.
