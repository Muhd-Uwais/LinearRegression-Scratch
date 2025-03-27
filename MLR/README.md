# Multiple Linear Regression (MLR) from Scratch

This repository implements **Multiple Linear Regression (MLR)** from scratch **without using scikit-learn**. It includes:
- Data generation
- Model training using **Gradient Descent**
- Model evaluation using **Mean Squared Error (MSE)**

## 📌 Equations Used

### 1️⃣ Multiple Linear Regression Equation
The equation for Multiple Linear Regression is:

$$ y = Xw + c $$

Where:
- $X$ is the **input feature matrix**
- $w$ is the **weight vector** (coefficients)
- $c$ is the **intercept** (bias term)
- $y$ is the **predicted output**

---
### 2️⃣ Mean Squared Error (MSE)
The Mean Squared Error is used as the **loss function**:

$$ \text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 $$

Where:
- $y_i$ is the **actual target value**
- $\hat{y}$ is the **predicted value**
- $n$ is the **number of data points**

---
### 3️⃣ Gradient Descent
To optimize **\( w \)** and **\( c \)**, we use **Gradient Descent**:

#### **Partial Derivatives**
$$ \frac{\partial J}{\partial w} = \frac{-2}{n} X^T (y - \hat{y}) $$
$$ \frac{\partial J}{\partial c} = \frac{-2}{n} \sum (y - \hat{y}) $$

#### **Updating Equations**
$$ w = w - \alpha \frac{\partial J}{\partial w} $$
$$ c = c - \alpha \frac{\partial J}{\partial c} $$

Where  $\alpha$  is the **learning rate**.

---
## 🛠 Code Structure

- `generate_data(n_samples, n_features)` → Generates **synthetic dataset**
- `compute_loss(y_true, y_pred)` → Computes **Mean Squared Error**
- `compute_gradients(X, y, w, c)` → Computes gradients for **\( w \)** and **\( c \)**
- `train_model(X, y, learning_rate, epochs)` → Runs **Gradient Descent** algorithm
- `predict(X, w, c)` → Predicts target values
- `plot_loss(loss_history)` → Plots **training loss**

---
## 🔧 Installation
```sh
pip install numpy matplotlib tqdm scikit-learn
```

---
## 🚀 Usage
```sh
python mlr.py
```

---
## 🎯 Example Output
```
Final parameters: w = [2.98, 1.99, 1.47], c = 3.95
Mean Squared Error: 2.85
```

---
## 📊 Visualization
- **Loss vs. Epochs:** Shows how loss decreases during training.
- **Predictions vs. Actual Data:** Compares model predictions with actual values.

📌 **This project is for educational purposes and demonstrates how Multiple Linear Regression works under the hood.** 🚀

