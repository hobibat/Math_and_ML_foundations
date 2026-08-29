# Module 01: Foundations & Univariate Linear Regression

## 1. Machine Learning Paradigms

* **Supervised Learning:** The algorithm learns a mapping function from input features $x$ to target labels $y$ using labeled training pairs $(x^{(i)}, y^{(i)})$.
  * **Regression:** Predicts a continuous, real-valued output (e.g., predicting house prices based on square footage).
  * **Classification:** Predicts discrete categories by determining decision boundaries between classes.
* **Unsupervised Learning:** The algorithm receives unlabeled data to discover underlying structure, groupings, or patterns without explicit targets.
  * **Clustering:** Groups data points into clusters based on geometric or statistical similarity.
  * **Anomaly Detection:** Identifies rare events or statistical outliers.
  * **Dimensionality Reduction:** Compresses feature space while preserving essential variance.

---

## 2. Univariate Linear Regression Model

### Mathematical Notation
* $x$: Input feature
* $y$: Target variable (ground truth)
* $\hat{y}$: Model prediction ($\hat{y} = f_{w, b}(x)$)
* $m$: Number of training examples
* $(w, b)$: Model parameters (weight/slope and bias/intercept)

### Hypothesis Function
$$\hat{y}^{(i)} = w x^{(i)} + b$$

---

## 3. The Cost Function (Mean Squared Error)

To measure how well the linear model fits the data, we compute the Mean Squared Error (MSE):

$$J(w, b) = \frac{1}{2m} \sum_{i=1}^{m} \left( \hat{y}^{(i)} - y^{(i)} \right)^2$$

* The $\frac{1}{2}$ term cancels out the exponent $2$ when computing derivatives during gradient updates.
* **Convexity:** Because MSE is a sum of squared linear terms, the cost surface $J(w, b)$ forms a strictly convex paraboloid ("bowl shape"). This guarantees **no local minima traps**—any local minimum is the global minimum.

---

## 4. Batch Gradient Descent

Gradient descent iteratively steps down the cost surface toward the global minimum:

### 1. Update Rules
$$w := w - \alpha \frac{\partial J(w, b)}{\partial w}$$
$$b := b - \alpha \frac{\partial J(w, b)}{\partial b}$$

### 2. Analytical Derivatives
$$\frac{\partial J(w, b)}{\partial w} = \frac{1}{m} \sum_{i=1}^{m} \left( \hat{y}^{(i)} - y^{(i)} \right) x^{(i)}$$

$$\frac{\partial J(w, b)}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} \left( \hat{y}^{(i)} - y^{(i)} \right)$$

### 3. Dynamics of the Learning Rate ($\alpha$)
* **$\alpha$ too small:** Convergence is extremely slow, requiring excessive iterations.
* **$\alpha$ too large:** Steps overshoot the minimum, potentially causing oscillation or divergence ($J \to \infty$).
* **At the minimum:** The partial derivatives equal $0$ ($\frac{\partial J}{\partial w} = 0$), so parameter updates naturally stop without adjusting $\alpha$.