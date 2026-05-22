# 2.1 Machine Learning Foundation

Welcome to the foundation of the Smart Air Force Warfighter. Before we can deploy advanced neural networks or autonomous drone wingmen, we must master the statistical algorithms that form the bedrock of artificial intelligence. In this lesson, we will explore how mathematical models learn from data to predict outcomes, classify threats, and optimize missions.

## Supervised vs. Unsupervised Learning

At the highest level, machine learning is split into two primary paradigms based on the data available to the algorithm:

* **Supervised Learning:** The algorithm is trained on a labeled dataset. It learns a mapping from inputs (features, like radar cross-section) to known outputs (labels, like "F-35" or "Civilian Airliner"). It acts as a student with an answer key.
* **Unsupervised Learning:** The algorithm is given unlabeled data and must find hidden structures or patterns on its own. It acts as an analyst grouping unidentified radar anomalies based on similarities in speed and altitude, without knowing what those anomalies are.

In this lesson, we focus entirely on **Supervised Learning** to build reliable classification models.

---

## Logistic Regression and LDA

**Scenario:** Predictive Maintenance. You are tasked with analyzing historical telemetry from aircraft components to predict engine failures before they happen.

### The Math: Logistic Regression

Despite its name, Logistic Regression is a classification algorithm. It uses a linear equation but passes the result through a **sigmoid function** to squash the output into a probability between 0.0 and 1.0.

The probability of engine failure given feature vector $x$ is:


$$P(y=1|x) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_1 + \dots + \beta_n x_n)}}$$

If $P > 0.5$, the model predicts a failure.

### The Math: Linear Discriminant Analysis (LDA)

Instead of modeling the probability directly, LDA models the distribution of the features for each class (Failed vs. Operational). It assumes the data for both classes share the same variance but have different means. It uses Bayes' Theorem to calculate the probability:

$$P(Y=k | X=x) = \frac{f_k(x) \pi_k}{\sum_{l=1}^{K} f_l(x) \pi_l}$$

Where $\pi_k$ is the prior probability of class $k$, and $f_k(x)$ is the Gaussian density function.

### Exercise: Predicting Component Failure (LINK)
---

## Support Vector Machines (SVM) and QDA

**Scenario:** Radar Cross-Section Classification. You need to classify incoming aircraft as friendly or adversarial based on non-linear radar cross-section profiles and speed.

### The Math: Support Vector Machine

An SVM seeks the optimal hyperplane that separates the classes with the maximum possible margin. The "support vectors" are the data points closest to the boundary.

The goal is to maximize the margin $\frac{2}{||w||}$ by solving the optimization problem:


$$\min_{w,b} \frac{1}{2} ||w||^2$$


Subject to the constraint that all points are classified correctly. To handle non-linear data, SVM uses the **Kernel Trick** to project data into higher dimensions where a linear cut becomes possible.

> **Insight:** Quadratic Discriminant Analysis (QDA) allows each class to have its own variance. This allows QDA to draw curved, non-linear boundaries naturally, making it highly effective for complex radar profiles where flight envelopes overlap in curves, not straight lines.

### Exercise: Classifying Radar Profiles
---

## Random Forests & The Bias-Variance Tradeoff

**Scenario:** Mission Success Prediction. You must predict the probability of mission success based on a massive matrix of weather, logistics, and intelligence data.

### The Math: Bias-Variance Tradeoff

This is the fundamental tension in machine learning. The expected error of any model can be broken down into three parts:


$$E[(y - \hat{f}(x))^2] = \text{Bias}(\hat{f}(x))^2 + \text{Var}(\hat{f}(x)) + \sigma^2$$

* **Bias:** Error from erroneous assumptions (e.g., assuming a relationship is linear when it is complex). High bias leads to **underfitting**.
* **Variance:** Error from sensitivity to small fluctuations in the training set. High variance leads to **overfitting** (memorizing the noise).
* **$\sigma^2$ (Irreducible Error):** The inherent noise in the data itself.

### Random Forests

A single Decision Tree is prone to high variance. A **Random Forest** fixes this through *bagging* (Bootstrap Aggregating). It trains hundreds of shallow decision trees on random subsets of data and averages their predictions. By averaging many high-variance models, the overall variance drops significantly while maintaining low bias.

### Exercise: Random Forest Generalization

---

## Knowledge Check

1. If an Intelligence Surveillance Reconnaissance (ISR) model achieves 99% accuracy on historical training data but only 60% accuracy on new incoming data, what specific problem from the bias-variance tradeoff is occurring?
2. Why would a Data Scientist choose an SVM with an RBF kernel over Logistic Regression for classifying radar cross-sections?
3. In the context of the Random Forest algorithm, explain how "bagging" reduces the overall variance of the model compared to a single decision tree.