# Mini-Batch Gradient Descent Regressor (From Scratch)

This project implements **Mini-Batch Gradient Descent (MBGD) for Linear Regression from scratch using NumPy**.
Instead of relying on built-in machine learning models, the optimization process is manually implemented to demonstrate how gradient descent updates model parameters.

The model updates weights and bias iteratively using **mini-batches of training samples**, providing a balance between the stability of batch gradient descent and the efficiency of stochastic gradient descent.

---

## Overview

Mini-Batch Gradient Descent is an optimization algorithm used to minimize a loss function by updating parameters using small subsets of the training data.

| Method                          | Description                              |
| ------------------------------- | ---------------------------------------- |
| Batch Gradient Descent          | Uses the entire dataset for each update  |
| Stochastic Gradient Descent     | Uses one sample per update               |
| **Mini-Batch Gradient Descent** | Uses a small batch of samples per update |

Mini-batch training generally leads to **faster convergence and smoother updates**.

---

## Implementation Details

The model is implemented using a custom class:

```python
class MBGDRegressor
```

### Parameters

| Parameter       | Description                                 |
| --------------- | ------------------------------------------- |
| `batch_size`    | Number of samples used per parameter update |
| `epochs`        | Number of passes through the training data  |
| `learning_rate` | Step size used during gradient updates      |

---

### Prediction

\[
\hat{y} = Xw + b
\]

---

### Gradient Computation

**Weight Gradient**

\[
\frac{-2}{n} X^T (y - \hat{y})
\]

**Bias Gradient**

\[
-2 \cdot \text{mean}(y - \hat{y})
\]

---

### Parameter Updates

**Weight Update**

\[
w = w - \eta \cdot \text{gradient}_w
\]

**Bias Update**

\[
b = b - \eta \cdot \text{gradient}_b
\]

---

### Where

- \(w\) = model weights (coefficients)  
- \(b\) = intercept (bias term)  
- \(X\) = feature matrix  
- \(y\) = true target values  
- \(\hat{y}\) = predicted values  
- \(n\) = number of samples in the mini-batch  
- \(\eta\) = learning rate

## Model Architecture

The regressor maintains two learnable parameters:

* **Weights (coefficients)**
* **Bias (intercept)**

Prediction is computed as:

```
y = X · w + b
```

---

## Evaluation Metrics

Model performance is evaluated using:

* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)
* R² Score
* Adjusted R² Score

Example output:

```
Mean Absolute Error: 47.8897
Mean Squared Error: 3323.3856
Root Mean Squared Error: 57.6488
R2 Score: 0.3553
Adjusted R2 Score: 0.3364
```

---

## Key Learning Points

This project demonstrates:

* Implementation of gradient descent from scratch
* Mini-batch optimization
* Linear regression training loops
* Vectorized gradient computation using NumPy
* Evaluation of regression models using standard metrics

---

## Technologies Used

* Python
* NumPy
* scikit-learn (for evaluation metrics)

---

## How to Run

Clone the repository:

```bash
git clone https://github.com/yourusername/MBGD-Regressor-From-Scratch
```

Install dependencies:

```bash
pip install numpy scikit-learn
```

Run the Python script or notebook.
