# Simple Linear Regression from Scratch

This repository implements **Simple Linear Regression (SLR)** from scratch without using `sklearn`. It includes data generation, model training using **Gradient Descent**, and model evaluation using **Mean Squared Error (MSE)**.

## Equations Used

### 1. Simple Linear Regression Equation
The equation for simple linear regression is:

$$ y = mX + c $$

where:
- $m$ is the slope (coefficient of $X$)
- $c$ is the intercept (bias term)
- $X$ is the input feature
- $y$ is the predicted output

### 2. Mean Squared Error (MSE)
The **Mean Squared Error** is used as the loss function:

$$ \text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 $$

where:
- $y_i$ is the actual target value
- $\hat{y}_i$ is the predicted value
- $n$ is the number of data points

### 3. Gradient Descent
To optimize $m$ and $c$, we use **Gradient Descent**:

#### Partial Derivatives:

$$ \frac{\partial J}{\partial m} = \frac{-2}{n} \sum X (y - \hat{y}) $$
$$ \frac{\partial J}{\partial c} = \frac{-2}{n} \sum (y - \hat{y}) $$

#### Updating Equations:

$$ m = m - \alpha \frac{\partial J}{\partial m} $$
$$ c = c - \alpha \frac{\partial J}{\partial c} $$

where $\alpha$ is the **learning rate**.

## Code Structure
- `generate_data(n_samples, test_size)`: Generates synthetic data
- `compute_loss(y_true, y_pred)`: Computes **Mean Squared Error**
- `compute_gradients(x, y, slope, intercept)`: Computes gradients for $m$ and $c$
- `predict(x, m, c)`: Predicts target values
- `fit(x, y, alpha, epochs)`: Runs **Gradient Descent** algorithm
- `plot_loss(loss_history)`: Plots training loss
- `plot_regression_line(X, y, m, c)`: Plots best-fit line

## Installation

```bash
pip install numpy matplotlib tqdm scikit-learn
```

## Usage

```bash
python slr.py
```

## Example Output

```
Final parameters: m = 2.53, c = 5.12
Mean Squared Error: 3.84
```

## Visualization
- **Loss vs. Epochs**: Shows how loss decreases over training iterations.
- **Regression Line**: Displays the best-fit line against actual data points.
