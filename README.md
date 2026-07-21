
````markdown
# 🔍 Logistic Regression Gradient Checking from Scratch

> A simple NumPy implementation of **Logistic Regression**, including the **cost function**, **analytical gradients**, and **numerical gradient checking** — built from scratch without using machine-learning libraries.


## 📌 Overview 

This project demonstrates how the gradients used in **Logistic Regression** can be calculated in two different ways and then compared to verify their correctness.
The project implements:

- 📉 Logistic Regression **cost function**
- 🧮 Analytical gradient calculation
- 🔢 Numerical gradient approximation using the **Central Difference Method**
- ✅ Gradient checking by comparing analytical and numerical gradients
- 📊 Calculation of the difference between the two approaches using the **L2 norm**

The main goal is to verify that the manually derived gradients are correctly implemented.


## 🎯 Purpose

In machine learning, gradient-based optimization methods such as **Gradient Descent** rely on accurate gradients to update model parameters.
For Logistic Regression, the parameters are:

- `w` → Weight vector
- `b` → Bias term

The gradient can be calculated analytically using the mathematical derivative of the cost function. However, implementing derivatives manually can sometimes lead to mistakes.
This project uses **numerical gradient checking** as a way to validate the analytical implementation.

The basic idea is:

```

Analytical Gradient
       │
       │ Compare
       ▼
Numerical Gradient
       │
       ▼
Small Difference?
       │
       ├── YES → ✅ Gradient implementation is likely correct
       │
       └── NO  → ❌ Check the derivative implementation

````


## 🧠 Logistic Regression

The model first calculates the linear combination:

$$
z = Xw + b
$$

Then applies the sigmoid function:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

The sigmoid function converts the model output into a value between `0` and `1`, which can be interpreted as the probability of belonging to the positive class.
The implementation is:

```python
def sigmoid(s):
    return 1 / (1 + np.exp(-s))
```


## 📉 Cost Function

The project uses the **Binary Cross-Entropy (Log Loss)** cost function:

$$
J(w,b) =
-\frac{1}{m}
\sum_{i=1}^{m}
\left[
y^{(i)}\log(g^{(i)})
+
(1-y^{(i)})\log(1-g^{(i)})
\right]
$$

where:

* `m` = number of training examples
* `y` = actual labels
* `g` = predicted probabilities

The cost function measures how well the Logistic Regression model's predictions match the true labels.

### Python Implementation

```python
def J(X, y, w, b):
    m = X.shape[0]
    g = sigmoid(np.dot(X, w) + b)

    return -(1 / m) * np.sum(
        y * np.log(g) + (1 - y) * np.log(1 - g)
    )
```


## 🧮 Analytical Gradient

The analytical gradients of the Logistic Regression cost function are calculated using:

$$
\frac{\partial J}{\partial w}
=============================

\frac{1}{m}X^T(g-y)
$$

and:

$$
\frac{\partial J}{\partial b}
=============================

\frac{1}{m}\sum_{i=1}^{m}(g-y)
$$

These are implemented in the `leftside()` function.

```python
def leftside(X, y, w, b):
    m = X.shape[0]
    g = sigmoid(np.dot(X, w) + b)

    dwj = (1 / m) * np.dot(X.T, (g - y))
    dbj = (1 / m) * np.sum(g - y)

    return dwj, dbj
```


## 🔢 Numerical Gradient

Instead of calculating the derivative directly, the numerical gradient approximates it using the **Central Difference Method**:

$$
\frac{\partial J}{\partial w_j}
\approx
\frac{
J(w_j+\epsilon)-J(w_j-\epsilon)
}{
2\epsilon
}
$$

where:

$$
\epsilon = 10^{-8}
$$

The same idea is applied to the bias `b`.

The `rightside()` function calculates these numerical approximations by slightly increasing and decreasing each parameter.

```text
              J(w + ε)
                 │
                 │
                 ▼
             Difference
                 │
                 ▲
                 │
              J(w - ε)

       Numerical Gradient
              ≈
   [J(w + ε) - J(w - ε)] / 2ε
```

This approach is computationally more expensive than using analytical gradients, but it is extremely useful for **debugging and validating gradient implementations**.


## 🔍 Gradient Checking

The main purpose of the project is to compare:

```text
Analytical Gradient
        vs.
Numerical Gradient
```

The difference between the two gradient vectors is calculated using the **L2 norm**:

```python
np.linalg.norm(dw_j - dw_j1)
```

For the bias, the absolute difference is calculated:

```python
abs(db_j - db_j1)
```

If the differences are sufficiently small, it provides evidence that the analytical gradient has been implemented correctly.

### Example Output

```text
[... ...] and [... ...] are dw
diff of them is: 1.23e-09

and 0.123456 and 0.123456 are db
diff is: 2.34e-10

the diffs are small
```

> 💡 **Key idea:** A small difference between the analytical and numerical gradients indicates that the derivative implementation is likely correct.


## 🧪 Dataset

For demonstration purposes, the project generates a synthetic dataset using NumPy.

```python
m, n = 200, 2

X = np.random.rand(m, n)
y = (np.sum(X, axis=1) > 1).astype(int)
```

The dataset contains:

* **200 samples**
* **2 features**
* **Binary target labels**

The labels are generated based on whether the sum of the two features is greater than `1`.

The model parameters are also initialized randomly:

```python
w = np.random.randn(n)
b = np.random.randn()
```


## 🛠️ Technologies Used

* 🐍 **Python**
* 🔢 **NumPy**
* 📐 Vectorized mathematical operations
* 🧮 Numerical differentiation
* 📊 Linear algebra

No machine-learning framework is required.


## 🚀 Possible Future Improvements

* [ ] Implement Gradient Descent
* [ ] Train the Logistic Regression model using the calculated gradients
* [ ] Add model predictions
* [ ] Calculate classification accuracy
* [ ] Visualize the synthetic dataset
* [ ] Visualize the decision boundary
* [ ] Compare analytical and numerical gradients for different values of `ε`
* [ ] Add numerical stability improvements to the sigmoid and cost functions
* [ ] Extend the implementation to multiclass classification


## 📚 Concepts Demonstrated

`Python` · `NumPy` · `Logistic Regression` · `Binary Cross-Entropy` · `Gradient Descent` · `Analytical Derivatives` · `Numerical Differentiation` · `Central Difference` · `Gradient Checking` · `Linear Algebra` · `Machine Learning`


## 💡 Why This Project?

Although this is a relatively small project, it demonstrates an important foundation in machine learning: understanding **what happens behind the scenes** rather than relying only on high-level libraries.

By implementing the mathematical components manually and validating the gradients numerically, this project provides a practical demonstration of the connection between:

> **Mathematics → Algorithms → Python Implementation → Validation**


## 👩‍💻 Author

**Anoosha Sameti**

Biomedical Engineering Undergraduate


### 📜 License

This project is intended for **educational and learning purposes**.

````
