# Digit-Classification-Neural-Network
Neural network made from scratch using just numpy and math!

## Overview

This project implements a fully connected neural network for digit classification on the MNIST dataset using only NumPy.
The focus is on understanding the math behind forward propagation, backpropagation, and gradient descent without relying on high-level frameworks like TensorFlow or PyTorch.

[You can see my network here!]([[url](https://www.kaggle.com/code/shrish34/neural-network-for-digit-classification)])

## Objectives

Build a neural network from first principles.

Implement forward pass, loss calculation, and backpropagation with only NumPy.

Document and connect the mathematical derivations (PDF) to the actual code implementation (Notebook).

Train the model to classify handwritten digits (0–9).

## Math Behind the Network

All mathematical foundations — from activation functions to gradient updates — are documented in Math For Neural Network.pdf

### Key steps:

#### 1. Forward Propagation

Linear transformation

Activation (ReLU / Softmax).

Loss Function

Cross-entropy loss between predicted probabilities and true labels.


#### 2. Backward Propagation

Gradients derived with the chain rule.

#### 3. Updating Parameters


## Implementation

The full implementation and experiments are contained in Neural Network for Digit Classification.ipynb

Features:

Vectorized forward and backward propagation.

ReLU activation for hidden layers, Softmax for output.

Gradient descent with learning rate as a hyperparameter.

Trains on 28×28 MNIST images (flattened into 784-dimensional vectors).

## Results

Achieves reasonable accuracy (~82%)

Training/validation curves are plotted in the notebook.

## Requirements

Python 3.9+

NumPy

Matplotlib

Jupyter Notebook (optional, for interactive exploration)
