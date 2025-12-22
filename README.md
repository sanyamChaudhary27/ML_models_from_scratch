# ML_models_from_scratch
Building machine Models from scratch in python, using numpy and pandas, it Enhances mathematical, statistical coding skills exponentially. 

## Linear Regression

Linear Regression is like a Hello World in ML world a fundamental supervised learning algorithm used for predicting a continuous dependent variable (y) based on one or more independent variables (X). This implementation builds the model entirely from scratch using **Python** and **NumPy**, providing a deep understanding of the underlying mathematics without relying on high-level libraries like Scikit-Learn.

### Mathematical Foundation

The model aims to find the best-fitting linear relationship between inputs and targets by minimizing the error.

**1. Hypothesis Function**
The relationship is modeled as a linear equation:
$$ h_\theta(x) = wx + b $$
Where:
*   $$ w $$ is the weight (slope)
*   $$ b $$ is the bias (intercept)
*   $$ x $$ is the input feature

**2. Cost Function (Mean Squared Error)**
To evaluate the model's performance, we use the Mean Squared Error (MSE) cost function, which measures the average squared difference between predicted and actual values:
$$ J(w, b) = \frac{1}{N} \sum_{i=1}^{N} (y_i - (wx_i + b))^2 $$

**3. Optimization (Gradient Descent)**
We minimize the cost function by iteratively updating the weights and bias using Gradient Descent. The gradients are calculated as:

$$ \frac{\partial J}{\partial w} = \frac{-2}{N} \sum_{i=1}^{N} x_i (y_i - y_{pred}) $$
$$ \frac{\partial J}{\partial b} = \frac{-2}{N} \sum_{i=1}^{N} (y_i - y_{pred}) $$

The parameters are updated using the learning rate ($$ \alpha $$):
$$ w = w - \alpha \frac{\partial J}{\partial w} $$
$$ b = b - \alpha \frac{\partial J}{\partial b} $$

### Implementation Details

The `LinearRegression` class is implemented with the following key components:

*   **`__init__(learning_rate, n_iterations)`**: Initializes the hyperparameters.
*   **`fit(X, y)`**: Training loop that executes gradient descent. It initializes parameters, computes predictions, calculates gradients, and updates weights/bias for `n_iterations`.
*   **`predict(X)`**: Outputs predictions for new data using the learned $$ w $$ and $$ b $$.

### Usage

```python
import numpy as np
import pandas as pd
from Linear_Regression import LinearRegression  # Import your class

# 1. Load Data
# X = ... (Features)
# y = ... (Target)

# 2. Initialize Model
# learning_rate: Controls step size (e.g., 0.01)
# n_iterations: Number of training epochs (e.g., 1000)
model = LinearRegression(learning_rate=0.01, n_iterations=1000)

# 3. Train the Model
model.fit(X, y)

# 4. Make Predictions
predictions = model.predict(X_test)

# 5. Output Parameters
print(f"Learned Weight: {model.weights}")
print(f"Learned Bias: {model.bias}")
```

### Key Learnings
Building this model from scratch reinforces several core machine learning concepts:
*   **Vectorization**: Using `numpy` for efficient matrix operations instead of explicit Python loops.
*   **Gradient Descent**: Understanding how partial derivatives guide the model toward the global minimum.
*   **Parameter Tuning**: Observing how `learning_rate` and `n_iterations` affect convergence.
