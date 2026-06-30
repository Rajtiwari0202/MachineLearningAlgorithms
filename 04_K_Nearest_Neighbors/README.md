# K-Nearest Neighbors From Scratch

This project implements K-Nearest Neighbors (KNN) from scratch using Python without relying on machine learning libraries.

KNN is a supervised machine learning algorithm used for classification and regression. This chapter focuses on classification.

## Concepts Covered

* Instance-Based Learning
* Lazy Learning
* Distance Calculation
* Euclidean Distance
* Choosing K
* Finding Nearest Neighbors
* Majority Voting
* Classification
* Prediction on Unseen Data

## Files

### notes.md

Contains theoretical notes and intuition behind K-Nearest Neighbors.

### distance_demo.py

A simple demonstration of distance calculation, nearest neighbors, and majority voting.

### knn_from_scratch.py

A reusable KNN class with fit(), predict_one(), and predict() methods.

## Example

Dataset:

* Low height, low weight -> child
* Medium height, medium weight -> teen
* High height, high weight -> adult

Prediction:

* A new person's height and weight are compared with all training examples
* The nearest K examples vote for the final class

## Future Improvements

* KNN Regression
* Feature Scaling
* Train/Test Split
* Accuracy Calculation
* Scikit-Learn Comparison
* Decision Boundary Visualization
