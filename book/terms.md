The TeachBooks template (built on Jupyter Book and Sphinx) includes a native `{glossary}` directive specifically for this purpose. This is vastly superior to a standard Markdown list because it automatically adds the terms to the book's overarching Index page, and allows you to cross-reference terms throughout the textbook using the `{term}` role.

Here is the properly formatted TeachBook/Sphinx Markdown for your glossary.

---

# Glossary of Core AI/ML Terminology

This glossary defines the foundational concepts introduced across the machine learning and artificial intelligence lessons.

```{glossary}
Activation Function
  The mathematical "gate" in an artificial neuron that determines whether, and how strongly, the neuron fires. It introduces non-linearity into neural networks. *(Examples: Sigmoid, ReLU).*

Artificial Neural Network (ANN)
  A computing system inspired by biological brains, consisting of interconnected nodes (neurons) organized in layers that automatically learn complex, non-linear representations of raw data.

Bagging (Bootstrap Aggregating)
  An ensemble machine learning technique that trains multiple models (like Decision Trees) on random subsets of data and averages their predictions to reduce variance and prevent overfitting.

Bias (Neural Networks)
  A learnable constant ($b$) added to a neuron's weighted sum. It acts as a baseline threshold, shifting the activation function left or right so the neuron can fire even if all input features are zero.

Bias (Statistical Error)
  The error introduced when a model makes overly simplistic assumptions about the data (e.g., assuming a relationship is a straight line when it is actually a curve). High bias leads to {term}`Underfitting`.

Bias-Variance Tradeoff
  The fundamental tension in machine learning where decreasing a model's false assumptions ({term}`Bias (Statistical Error)`) typically increases its dangerous sensitivity to noise ({term}`Variance (Statistical Error)`), and vice versa. 

Confusion Matrix
  A performance measurement table that visualizes exactly how a classification model succeeds or fails, categorizing predictions into True Positives, True Negatives, False Positives, and False Negatives. Crucial for weighing operational risks.

Decision Tree
  A supervised learning model that classifies data by recursively splitting it into branches based on the feature that best purifies the resulting groups (often measured by {term}`Gini Impurity`).

Feedforward Architecture
  A neural network design where information moves in only one direction—from the input layer, through hidden layers, directly to the output layer—without looping back.

Gini Impurity
  A mathematical metric used by Decision Trees to measure how "mixed" or impure a node of data is. A score of 0 means perfect purity (the node contains only one class).

Hidden Layer
  A layer of artificial neurons situated between the input and output layers of a neural network. These layers are responsible for learning abstract, hidden features in the data.

Hyperplane
  The mathematical decision boundary drawn by a Support Vector Machine to separate classes. In 2D space, it is a line; in 3D space, it is a flat plane.

Irreducible Error
  The inherent noise, randomness, or "fog of war" in a dataset that no machine learning model can ever predict or eliminate.

Kernel Trick
  A mathematical technique used by Support Vector Machines (SVMs) to project non-linear, complex data into higher dimensions where it can be easily sliced by a flat hyperplane.

Laplace Smoothing (Add-One Smoothing)
  A mathematical technique used in Naive Bayes to prevent the formula from collapsing to zero when the model encounters a feature (like a new vocabulary word) it has never seen before.

Likelihood
  In Bayes' Theorem, the probability of observing a specific piece of evidence assuming that a particular class is true.

Logistic Regression
  A foundational classification algorithm that passes a linear equation through a {term}`Sigmoid Function` to predict the probability of a binary outcome.

Naive Assumption (Independence Assumption)
  The core (and mathematically flawed) assumption in Naive Bayes that every feature in a dataset is completely independent of every other feature.

Overfitting (High Variance)
  A failure state where a model becomes so overly complex that it memorizes the random noise in its training data rather than learning the underlying tactical rules. It performs perfectly in training but fails on unseen data.

Perceptron
  The simplest historical form of an artificial neuron. It takes multiple inputs, calculates a weighted sum, adds a bias, and passes the result through a hard step-function to output a 1 or a 0.

Posterior
  In Bayes' Theorem, the final, updated probability of a class being true *after* factoring in the new evidence.

Prior
  In Bayes' Theorem, the baseline, historical probability of a class occurring *before* any new evidence is observed.

Random Forest
  A powerful ensemble model that builds hundreds of shallow Decision Trees and averages their predictions to achieve high accuracy while avoiding the overfitting trap of single trees.

ReLU (Rectified Linear Unit)
  A highly efficient and popular activation function used in deep learning hidden layers. It outputs the raw input directly if it is positive, and outputs zero if it is negative.

Sigmoid Function
  A mathematical function that squashes any real number into a valid probability value bounded strictly between 0.0 and 1.0. 

Supervised Learning
  A machine learning paradigm where the algorithm is trained on historical data that already includes the correct answers (labels). It learns to map inputs to those known outputs.

Underfitting (High Bias)
  A failure state where a model is too rigid or simple to learn the underlying patterns in the training data, resulting in poor predictive performance across the board.

Unsupervised Learning
  A machine learning paradigm where the algorithm is given raw, unlabeled data and must autonomously discover hidden structures, clusters, or patterns on its own.

Variance (Statistical Error)
  The error introduced when a model is overly sensitive to tiny fluctuations or noise in the training data. High variance leads to {term}`Overfitting (High Variance)`.

Weights
  The learnable parameters ($w_i$) in a neural network that determine the importance or influence of a specific input feature on the final prediction.

```

---

### TeachBook Cross-Referencing Tip

Because we used the `{glossary}` directive, you can now link back to these definitions from anywhere else in your textbook.

If you are writing Lesson 2.4 and want to mention Overfitting, you simply write:
`This leads to the danger of {term}Overfitting (High Variance).`

TeachBook will automatically generate a hyperlink directly back to this glossary page.

Are there any specific CSS styles you want applied to this list, or shall we move on to drafting Lesson 2.4?