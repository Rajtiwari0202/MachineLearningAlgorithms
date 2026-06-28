# Multiple Linear Regression From Scratch

This project implements Multiple Linear Regression from scratch using Python and Gradient Descent without relying on machine learning libraries.

Multiple Linear Regression is an extension of Linear Regression where the model learns from more than one input feature.

## Concepts Covered

* Multiple Input Features
* Prediction Function
* Weights and Bias
* Error Calculation
* Mean Squared Error (MSE)
* Cost Function
* Partial Derivatives
* Gradient Descent
* Model Training
* Prediction on Unseen Data

## Files

### notes.md

Contains theoretical notes and intuition behind Multiple Linear Regression.

### gradient_descent_demo.py

A simple implementation of Gradient Descent on a small dataset with multiple features.

### multiple_linear_regression_from_scratch.py

A reusable Multiple Linear Regression class with fit() and predict() methods.

## Example

Dataset:

* 1 hour studied, 1 assignment completed -> 30 marks
* 2 hours studied, 1 assignment completed -> 50 marks
* 3 hours studied, 2 assignments completed -> 80 marks
* 4 hours studied, 2 assignments completed -> 100 marks

Learned Parameters:

* weights are learned for each feature
* bias is learned separately

Prediction:

* 5 hours studied, 3 assignments completed -> predicted marks

## Future Improvements

* NumPy Vectorization
* Feature Scaling
* Train/Test Split
* Scikit-Learn Comparison
* Visualization
