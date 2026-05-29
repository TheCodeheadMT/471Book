<div style=" background-color: #adadad; color: #000000; padding: 15px 20px; border-radius: 8px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif; border: 1px solid #444d56; line-height: 1.5; "> 

  <!-- Modified Title Block -->
  <div style="font-size: 3em; font-weight: 600; margin-bottom: 15px; border-bottom: 1px solid #777; padding-bottom: 5px;">
    L27: ML Foundations
  </div>

  <!-- Lesson Summary -->
  <div style=" background-color: #2f363d; color: #f1f8ff; padding: 15px 20px; border-radius: 8px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif; border: 1px solid #444d56; line-height: 1.5; "> 
    Welcome to the foundation of the Smart Air Force Warfighter. Before we can deploy advanced neural networks or autonomous drone wingmen, we must master the statistical algorithms that form the bedrock of artificial intelligence. In this lesson, we will explore how mathematical models learn from data to predict outcomes, classify threats, and optimize missions. 
  </div> 

  <!-- Lesson Objectives -->
  * Lesson Objective #1 
  * Lesson Objective #2 
  * Lesson Objective #3  
</div>
<br>

## Supervised vs. Unsupervised Learning

At the highest level, machine learning is split into two primary paradigms based on the data available to the algorithm:

* **Supervised Learning:** The algorithm is trained on a labeled dataset. It learns a mapping from inputs (features, like radar cross-section) to known outputs (labels, like "F-35" or "Civilian Airliner"). It acts as a student with an answer key.
* **Unsupervised Learning:** The algorithm is given unlabeled data and must find hidden structures or patterns on its own. It acts as an analyst grouping unidentified radar anomalies based on similarities in speed and altitude, without knowing what those anomalies are.

In this lesson, we focus entirely on **Supervised Learning** to build reliable classification models.

<br>
<hr width="100%" size="4" color="black">

## Logistic Regression, LDA, & QDA

**Scenario:** Predictive Maintenance. You are tasked with analyzing historical telemetry from aircraft components to predict engine failures before they happen.

### The Math: Logistic Regression

Despite its name, Logistic Regression is a classification algorithm. It uses a linear equation but passes the result through a **sigmoid function** to squash the output into a probability between 0.0 and 1.0.

The probability of engine failure given feature vector $x$ is:


$$P(y=1|x) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_1 + \dots + \beta_n x_n)}}$$

If $P > 0.5$, the model predicts a failure.

>
> **Example 27.1 - Logistic Regression**
>
>Let's predict engine failure based on a single feature: Vibration ($x_1$).
>Suppose our trained model has an intercept $\beta_0 = -5$ and a vibration weight $\beta_1 = 0.5$.
>
>We receive a new sensor reading of $12\text{ Hz}$ vibration ($x_1 = 12$).
>1. Calculate the linear sum: $z = -5 + (0.5 \times 12) = -5 + 6 = 1$
>2. Apply the sigmoid function: $P = \frac{1}{1 + e^{-1}}$
>3. Since $e^{-1} \approx 0.367$, the probability is $P = \frac{1}{1 + 0.367} = \frac{1}{1.367} \approx 0.73$
>
>**Result:** The model predicts a 73% probability of engine failure. Since $0.73 > 0.5$, the mechanic is alerted.
>

### The Math: Linear Discriminant Analysis (LDA)

Instead of modeling the probability directly, LDA models the distribution of the features for each class (Failed vs. Operational). It assumes the data for both classes share the same variance but have different means. It uses Bayes' Theorem to calculate the probability:

$$P(Y=k | X=x) = \frac{f_k(x) \pi_k}{\sum_{l=1}^{K} f_l(x) \pi_l}$$

Where $\pi_k$ is the prior probability of class $k$, and $f_k(x)$ is the Gaussian density function.

> **Example 27.2 - LDA**
> Suppose 15% of engines fail historically ($\pi_{fail} = 0.15$), and 85% are operational ($\pi_{op} = 0.85$).
>
> We read a high temperature of $100^\circ\text{C}$ ($x = 100$). Based on historical distributions, the probability density of seeing $100^\circ\text{C}$ in a failing engine is $f_{fail}(100) = 0.04$, and in an operational engine is $f_{op}(100) = 0.01$.
> 1. Numerator (Failure score): $0.04 \times 0.15 = 0.006$
> 2. Numerator (Operational score): $0.01 \times 0.85 = 0.0085$
> 3. Total sum (Denominator): $0.006 + 0.0085 = 0.0145$
> 4. Final Probability of Failure: $\frac{0.006}{0.0145} \approx 0.413$
> 
> 
> **Result:** Despite the high temperature, the overwhelming prior probability that engines are usually fine drags the failure probability down to 41.3%. The engine is flagged for monitoring, but not immediately grounded.

Check out the [**Statitical ML Simulators**](https://thecodeheadmt.github.io/CS471/StatisticalML/index.html) to better understand how Linear Discriminant Analysis works.

### The Math: Quadratic Discriminant Analysis

LDA assumes all classes share the exact same variance. However, in the real battlespace, a drone swarm's flight envelope looks completely different from a cargo plane's. **QDA** solves this by giving every single class $k$ its own covariance matrix ($\Sigma_k$).

Because the variances no longer mathematically cancel out between classes, the resulting decision boundary is a curve (quadratic) rather than a straight line. The model classifies an observation $x$ by calculating a discriminant score $\delta_k(x)$ for each class and picking the highest one:

$$\delta_k(x) = -\frac{1}{2} \log |\Sigma_k| - \frac{1}{2} (x - \mu_k)^T \Sigma_k^{-1} (x - \mu_k) + \log \pi_k$$

For a single feature (1D data), the formula simplifies to:

$$\delta_k(x) = -\frac{1}{2} \ln(\sigma_k^2) - \frac{(x - \mu_k)^2}{2\sigma_k^2} + \ln(\pi_k)$$

> **Example 27.3 - QDA**
> Assume both Cargo Planes and Drone Swarms average 300 knots ($\mu = 300$), and they are equally common ($\pi = 0.5$). However, Cargo speed variance is tight ($\sigma^2_{Cargo} = 100$), while Drone speed variance is chaotic ($\sigma^2_{Drone} = 2500$).
> Radar picks up a target flying at $x = 320$ knots. Let's calculate the QDA scores (Note: $\ln(0.5) \approx -0.69$):
> 1. **Cargo Score:** > $\delta_C(320) = -0.5 \ln(100) - \frac{(320 - 300)^2}{2 \times 100} - 0.69$
> $\delta_C(320) = -2.30 - \frac{400}{200} - 0.69 = -2.30 - 2.00 - 0.69 = \mathbf{-4.99}$
> 2. **Drone Score:**
> $\delta_D(320) = -0.5 \ln(2500) - \frac{(320 - 300)^2}{2 \times 2500} - 0.69$
> $\delta_D(320) = -3.91 - \frac{400}{5000} - 0.69 = -3.91 - 0.08 - 0.69 = \mathbf{-4.68}$
> 
> 
> **Result:** Since $-4.68 > -4.99$, the model classifies the target as a **Drone Swarm**. Even though 320 knots is close to the Cargo average, QDA mathematically realizes that a 20-knot deviation is highly suspicious for a stable cargo flight, but completely normal for an erratic drone.

Check out the **[Statistical ML Simulators](https://thecodeheadmt.github.io/CS471/StatisticalML/index.html)** to better understand how Quadratic Discriminant Analysis works.

<br>
<hr width="100%" size="4" color="black"> 

## Support Vector Machines (SVM)

**Scenario:** Radar Cross-Section Classification. You need to classify incoming aircraft as friendly or adversarial based on non-linear radar cross-section profiles and speed.

### The Math: Support Vector Machine

An SVM seeks the optimal hyperplane that separates the classes with the maximum possible margin. The "support vectors" are the data points closest to the boundary.

The goal is to maximize the margin $\frac{2}{||w||}$ by solving the optimization problem:


$$\min_{w,b} \frac{1}{2} ||w||^2$$


Subject to the constraint that all points are classified correctly. To handle non-linear data, SVM uses the **Kernel Trick** to project data into higher dimensions where a linear cut becomes possible.


> **Example 27.3 - SVM Margin**
> Imagine we mapped friendly and adversarial radar returns onto a 2D grid. The SVM found the optimal dividing line with a weight vector $w = [3, 4]$.
> 1. Calculate the magnitude of the weight vector ($||w||$): $\sqrt{3^2 + 4^2} = \sqrt{9 + 16} = \sqrt{25} = 5$
> 2. Calculate the margin width: $\frac{2}{||w||} = \frac{2}{5} = 0.4$
> 
> 
> **Result:** The mathematical margin of $0.4$ represents the physical "buffer zone" in our feature space. A wider margin means the model is more confident and less likely to misclassify a slightly anomalous friendly radar blip as an adversary.
>
> **Insight:** Quadratic Discriminant Analysis (QDA) allows each class to have its own variance. This allows QDA to draw curved, non-linear boundaries naturally, making it highly effective for complex radar profiles where flight envelopes overlap in curves, not straight lines.

Check out the [**Statitical ML Simulators**](https://thecodeheadmt.github.io/CS471/StatisticalML/index.html) to better understand how a Support Vector Machine works.

<br>
<hr width="100%" size="4" color="black">

## Bias-Variance Tradeoff, Decision Trees, & Random Forests  
**Scenario:** Mission Success Prediction. You must predict the probability of mission success based on a massive matrix of weather, logistics, and intelligence data.

### The Math: Bias-Variance Tradeoff

This is the fundamental tension in machine learning. The expected error of any model can be broken down into three parts:

$$E[(y - \hat{f}(x))^2] = \text{Bias}(\hat{f}(x))^2 + \text{Var}(\hat{f}(x)) + \sigma^2$$

* **Bias:** Error from erroneous assumptions (e.g., assuming a relationship is linear when it is complex). High bias leads to **underfitting**.
* **Variance:** Error from sensitivity to small fluctuations in the training set. High variance leads to **overfitting** (memorizing the noise).
* **$\sigma^2$ (Irreducible Error):** The inherent noise in the data itself.

> **Example - Bias vs. Variance**
> Let's say the inherent, unpreventable "fog of war" noise ($\sigma^2$) in our mission data causes a baseline error of $2\%$. Our overall model error is $10\%$.
> 1. **Underfitting (High Bias):** A simple linear model might miss the complex reality of warfare. Its bias squared is $7\%$, and its variance is $1\%$. Total error $= 7\% + 1\% + 2\% = 10\%$.
> 2. **Overfitting (High Variance):** We switch to a single massive Decision Tree. It perfectly fits the training data, dropping bias to $1\%$. However, it memorizes random historical anomalies, spiking its variance to $7\%$. Total error $= 1\% + 7\% + 2\% = 10\%$.
> 
> 
> **Result:** Lowering bias almost always raises variance. The goal of ensemble methods is to find the "sweet spot" in the middle.

Check out the [**Statitical ML Simulators**](https://thecodeheadmt.github.io/CS471/StatisticalML/index.html) to better understand how Random Forests works.



### The Math: Decision Trees & Gini Impurity

A Decision Tree learns by recursively splitting data based on the feature that best separates the target classes (e.g., Success vs. Failure). To find the "best" split, the algorithm calculates the **Gini Impurity** of a node. A Gini score of $0$ means the node is perfectly pure (contains only one class).

For a node containing classes $C$, with the probability $p_i$ of an item belonging to class $i$, the Gini Impurity is:


$$Gini = 1 - \sum_{i=1}^{C} p_i^2$$

> **Example 27.4 - Decision Tree Splitting**
> A planning node contains 10 past missions: 6 Successes and 4 Failures.
> 1. Calculate the Root Node Impurity:
> $Gini_{root} = 1 - ( (0.6)^2 + (0.4)^2 ) = 1 - (0.36 + 0.16) = 1 - 0.52 = 0.48$
> 2. We split the data using a rule: "Was the Weather Clear?"
> * **Left Node (Clear Weather):** 5 missions (all 5 are Successes).
> $Gini_{left} = 1 - ( (1.0)^2 + (0.0)^2 ) = 1 - 1 = 0.0$ (Perfectly pure!)
> * **Right Node (Bad Weather):** 5 missions (1 Success, 4 Failures).
> $Gini_{right} = 1 - ( (0.2)^2 + (0.8)^2 ) = 1 - (0.04 + 0.64) = 1 - 0.68 = 0.32$
> 
> 
> 
> 
> **Result:** The weighted average impurity of the two child nodes is $\frac{5}{10}(0) + \frac{5}{10}(0.32) = 0.16$. Since $0.16 < 0.48$, splitting by "Weather" significantly reduced impurity. The algorithm keeps this split!

### Random Forests

A single Decision Tree is incredibly prone to high variance. If left unchecked, it will grow deep enough to achieve a Gini Impurity of $0$ on every leaf, memorizing the training data completely.

A **Random Forest** fixes this through *bagging* (Bootstrap Aggregating). It trains hundreds of shallow decision trees on random subsets of data and averages their predictions. By averaging many high-variance models, the overall variance drops significantly while maintaining low bias.

<br>
<hr width="100%" size="4" color="black">

## Summary Infographic
![ML Foundations](../../figures/L27_infographic.png "ML Foundations")

<br>
<hr width="100%" size="4" color="black">

## Knowledge Check

1. If an Intelligence Surveillance Reconnaissance (ISR) model achieves 99% accuracy on historical training data but only 60% accuracy on new incoming data, what specific problem from the {term}`Bias-Variance Tradeoff` is occurring?

2. Why would a Data Scientist choose an SVM with an RBF kernel over Logistic Regression for classifying radar cross-sections?

3. In the context of the Random Forest algorithm, explain how {term}`Bagging (Bootstrap Aggregating)` reduces the overall variance of the model compared to a single decision tree.

```{code-cell} ipython3
:tags: [remove-input]

from jupyterquiz import display_quiz
display_quiz("questions.json")
