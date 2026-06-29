# Logistic Regression From Scratch

This project implements Logistic Regression from scratch using Python and Gradient Descent without relying on machine learning libraries.

Logistic Regression is used for classification problems where the output belongs to a category, usually 0 or 1.

## Concepts Covered

* Binary Classification
* Linear Function
* Sigmoid Function
* Probability Prediction
* Error Calculation
* Binary Cross-Entropy Loss
* Cost Function
* Gradients
* Gradient Descent
* Model Training
* Class Prediction on Unseen Data

## Files

### notes.md

Contains theoretical notes and intuition behind Logistic Regression.

### gradient_descent_demo.py

A simple implementation of Gradient Descent on a small binary classification dataset.

### logistic_regression_from_scratch.py

A reusable Logistic Regression class with fit(), predict_probability(), and predict() methods.

## Example

Dataset:

* 1 hour studied, 40 attendance -> fail
* 2 hours studied, 50 attendance -> fail
* 3 hours studied, 60 attendance -> pass
* 4 hours studied, 70 attendance -> pass
* 5 hours studied, 80 attendance -> pass

Learned Parameters:

* weights are learned for each feature
* bias is learned separately

Prediction:

* 3.5 hours studied, 65 attendance -> predicted probability and class

## Future Improvements

* NumPy Vectorization
* Feature Scaling
* Train/Test Split
* Scikit-Learn Comparison
* Accuracy, Precision, Recall, and F1 Score
* Decision Boundary Visualization
