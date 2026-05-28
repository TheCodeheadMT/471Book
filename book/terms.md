
# AI/ML Terminology

This glossary defines the foundational concepts introduced across the machine learning and artificial intelligence lessons.

```{glossary}
1D-Convolutional Neural Network (1D-CNN)
  A deep learning architecture highly specialized for processing one-dimensional sequence data, such as raw time-series radio frequency (RF) signals in Electronic Warfare.

Activation Function
  The mathematical "gate" in an artificial neuron that determines whether, and how strongly, the neuron fires. It introduces non-linearity into neural networks. *(Examples: Sigmoid, ReLU).*

Artificial Neural Network (ANN)
  A computing system inspired by biological brains, consisting of interconnected nodes (neurons) organized in layers that automatically learn complex, non-linear representations of raw data.

Autoencoder
  An unsupervised neural network architecture consisting of an encoder and decoder, trained to reconstruct its own input. Highly effective for detecting zero-day cyber anomalies by measuring reconstruction failure.

Backpropagation
  The core algorithm used to train neural networks. It calculates the gradient (error) of the loss function with respect to every weight in the network, mathematically working backward from the output layer to the input layer.

Bagging (Bootstrap Aggregating)
  An ensemble machine learning technique that trains multiple models (like Decision Trees) on random subsets of data and averages their predictions to reduce variance and prevent overfitting.

Bias (Neural Networks)
  A learnable constant ($b$) added to a neuron's weighted sum. It acts as a baseline threshold, shifting the activation function left or right so the neuron can fire even if all input features are zero.

Bias (Statistical Error)
  The error introduced when a model makes overly simplistic assumptions about the data (e.g., assuming a relationship is a straight line when it is actually a curve). High bias leads to {term}`Underfitting (High Bias)`.

Bias-Variance Tradeoff
  The fundamental tension in machine learning where decreasing a model's false assumptions ({term}`Bias (Statistical Error)`) typically increases its dangerous sensitivity to noise ({term}`Variance (Statistical Error)`), and vice versa. 

Confusion Matrix
  A performance measurement table that visualizes exactly how a classification model succeeds or fails, categorizing predictions into True Positives, True Negatives, False Positives, and False Negatives. Crucial for weighing operational risks.

Decoder
  The second half of an Autoencoder that attempts to decompress the latent space bottleneck back into the original data format.

Decision Tree
  A supervised learning model that classifies data by recursively splitting it into branches based on the feature that best purifies the resulting groups (often measured by {term}`Gini Impurity`).

Deep Learning
  A subset of machine learning utilizing neural networks with many hidden layers (deep architectures) capable of autonomously engineering complex, hierarchical features from raw, unstructured data (like images or RF signals).

Dropout
  A regularization technique where a random percentage of neurons in a layer are temporarily disabled during each training step. This prevents the network from relying on specific neurons to memorize noise, forcing it to learn robust, generalized features.

Early Stopping
  A regularization technique that monitors the validation loss during training. It automatically halts the training process when the model stops improving on unseen data, saving the optimal weights before overfitting occurs.

Encoder
  The first half of an Autoencoder that compresses raw input data down into a mathematically dense bottleneck.

Epoch
  One complete pass of the entire training dataset through the neural network during the training phase.

Feedforward Architecture
  A neural network design where information moves in only one direction—from the input layer, through hidden layers, directly to the output layer—without looping back.

Gini Impurity
  A mathematical metric used by Decision Trees to measure how "mixed" or impure a node of data is. A score of 0 means perfect purity (the node contains only one class).

Gradient
  A mathematical vector of partial derivatives representing the direction of steepest ascent for the loss function.

Gradient Descent
  The optimization algorithm used to minimize a network's loss. It updates the network's weights iteratively by taking small mathematical steps in the opposite direction of the {term}`Gradient`.

Hidden Layer
  A layer of artificial neurons situated between the input and output layers of a neural network. These layers are responsible for learning abstract, hidden features in the data.

Hierarchical Feature Learning
  The process by which deep neural networks autonomously learn simple concepts (like lines and edges) in early layers and mathematically combine them into complex tactical concepts (like vehicles or radar structures) in deeper layers.

Hyperplane
  The mathematical decision boundary drawn by a Support Vector Machine to separate classes. In 2D space, it is a line; in 3D space, it is a flat plane.

Irreducible Error
  The inherent noise, randomness, or "fog of war" in a dataset that no machine learning model can ever predict or eliminate.

Kernel Trick
  A mathematical technique used by Support Vector Machines (SVMs) to project non-linear, complex data into higher dimensions where it can be easily sliced by a flat hyperplane.

Laplace Smoothing (Add-One Smoothing)
  A mathematical technique used in Naive Bayes to prevent the formula from collapsing to zero when the model encounters a feature (like a new vocabulary word) it has never seen before.

Latent Space (Bottleneck)
  The highly compressed, mathematical representation of data found in the middle layer connecting an encoder and decoder within an Autoencoder.

Learning Rate
  A hyperparameter ($\alpha$) that dictates how large of a step the network takes during {term}`Gradient Descent`. If it is too large, the network overcorrects erratically; if it is too small, learning is painfully slow.

Likelihood
  In Bayes' Theorem, the probability of observing a specific piece of evidence assuming that a particular class is true.

Logistic Regression
  A foundational classification algorithm that passes a linear equation through a {term}`Sigmoid Function` to predict the probability of a binary outcome.

Loss Function (Cost Function)
  A mathematical function that evaluates how far off a network's predictions are from the true targets. The network's entire training goal is to minimize this value.

Mean Squared Error (MSE)
  A common loss function used for regression tasks (predicting continuous numbers) that calculates the average squared difference between the predicted values and the actual target values.

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

Reconstruction Error
  The mathematical difference (often measured via MSE) between an original input and an Autoencoder's attempted reconstruction of it. Spikes in this error trigger anomaly alerts.

Regularization
  A set of techniques (like {term}`Dropout` or {term}`Early Stopping`) used to mathematically penalize complex models to prevent them from overfitting the training data.

ReLU (Rectified Linear Unit)
  A highly efficient and popular activation function used in deep learning hidden layers. It outputs the raw input directly if it is positive, and outputs zero if it is negative.

Sigmoid Function
  A mathematical function that squashes any real number into a valid probability value bounded strictly between 0.0 and 1.0. 

Supervised Learning
  A machine learning paradigm where the algorithm is trained on historical data that already includes the correct answers (labels). It learns to map inputs to those known outputs.

Transfer Learning
  A rapid deployment technique where a massive neural network, pre-trained on millions of generic data points, has its base layers "frozen" and its final classification head re-trained for a specific tactical domain.

Underfitting (High Bias)
  A failure state where a model is too rigid or simple to learn the underlying patterns in the training data, resulting in poor predictive performance across the board.

Unsupervised Learning
  A machine learning paradigm where the algorithm is given raw, unlabeled data and must autonomously discover hidden structures, clusters, or patterns on its own.

Variance (Statistical Error)
  The error introduced when a model is overly sensitive to tiny fluctuations or noise in the training data. High variance leads to {term}`Overfitting (High Variance)`.

Weights
  The learnable parameters ($w_i$) in a neural network that determine the importance or influence of a specific input feature on the final prediction.

```