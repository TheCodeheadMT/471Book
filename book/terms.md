
# AI/ML Terminology

This glossary defines the foundational concepts introduced across the machine learning and artificial intelligence lessons.

```{glossary}
1D-Convolutional Neural Network (1D-CNN)
  A deep learning architecture highly specialized for processing one-dimensional sequence data, such as raw time-series radio frequency (RF) signals in Electronic Warfare.

Action (ReAct)
  The step in the ReAct loop where the agent executes a specific, structured tool call (e.g., querying a database or calling an API) to interact with the external environment.

Activation Function
  The mathematical "gate" in an artificial neuron that determines whether, and how strongly, the neuron fires. It introduces non-linearity into neural networks. *(Examples: Sigmoid, ReLU).*

Adjudicator Agent
  In multi-agent wargaming, an impartial referee AI that receives proposed actions from competing agents (Red/Blue teams), evaluates them against simulated physical rules or probabilities, and updates the shared environment.

Adversarial Evasion
  A deployment-phase attack where an adversary subtly alters the physical environment or input data (e.g., using digital static or specialized camouflage) to mathematically force an already-trained AI model to make a highly confident, incorrect prediction.

Agent
  A system that processes information and produces an output from an input. It perceives its environment and takes actions to achieve a specific goal.

Agentic AI
  An AI system endowed with agency—the capacity to autonomously plan, reason, use tools, and take actions over multiple steps to achieve a complex goal, rather than simply reacting to a single prompt.

Algorithmic Bias
  A failure mode where an AI system produces systematically prejudiced or skewed results, usually caused by imbalanced, incomplete, or historically biased training data.

API (Application Programming Interface)
  A set of protocols that allows different software applications to communicate. In AI, APIs are used to send data (like raw text) to powerful cloud-hosted models (like LLMs) and securely receive the processed output.

Artificial Intelligence (AI)
  The study of how to make computers do things at which, at the moment, people are better (e.g., rapid pattern recognition, natural language processing, and complex decision-making).

Artificial Neural Network (ANN)
  A computing system inspired by biological brains, consisting of interconnected nodes (neurons) organized in layers that automatically learn complex, non-linear representations of raw data.

Autoencoder
  An unsupervised neural network architecture consisting of an encoder and decoder, trained to reconstruct its own input. Highly effective for detecting zero-day cyber anomalies by measuring reconstruction failure.

Backpropagation
  The core algorithm used to train neural networks. It calculates the gradient (error) of the loss function with respect to every weight in the network, mathematically working backward from the output layer to the input layer.

Bagging (Bootstrap Aggregating)
  An ensemble machine learning technique that trains multiple models (like Decision Trees) on random subsets of data and averages their predictions to reduce variance and prevent overfitting.

Base Model
  A foundational Large Language Model that has completed pre-training but has not yet undergone instruct-tuning. It excels at predicting the next word and understanding language, but cannot reliably follow commands or act as an assistant.

Bias (Neural Networks)
  A learnable constant ($b$) added to a neuron's weighted sum. It acts as a baseline threshold, shifting the activation function left or right so the neuron can fire even if all input features are zero.

Bias (Statistical Error)
  The error introduced when a model makes overly simplistic assumptions about the data (e.g., assuming a relationship is a straight line when it is actually a curve). High bias leads to {term}`Underfitting (High Bias)`.

Bias-Variance Tradeoff
  The fundamental tension in machine learning where decreasing a model's false assumptions ({term}`Bias (Statistical Error)`) typically increases its dangerous sensitivity to noise ({term}`Variance (Statistical Error)`), and vice versa. 

Cascading Hallucination
  A failure mode unique to multi-agent systems where one agent hallucinates a fact or capability, and other agents subsequently accept that hallucination as truth, rapidly derailing the entire workflow or simulation.

Connectionism
  A subdiscipline of AI popularized in the 1980s that abandoned explicit, programmed rules in favor of artificial neural networks that learn from training examples.

Confusion Matrix
  A performance measurement table that visualizes exactly how a classification model succeeds or fails, categorizing predictions into True Positives, True Negatives, False Positives, and False Negatives. Crucial for weighing operational risks.

Context Window
  The hard mathematical limit on the maximum number of tokens (words/sub-words) a Large Language Model can process in a single request. In agentic AI, this serves as the agent's Short-Term Memory.

Convolution
  A mathematical operation where a small matrix (filter/kernel) slides across an input image, multiplying and summing pixel values to detect specific visual features like edges, shapes, or textures.

Convolutional Neural Network (CNN)
  A specialized deep learning architecture designed to process grid-like data (such as images). It uses sliding filters to learn spatial hierarchies of features, making it highly effective for visual ISR and target recognition tasks.

Cosine Similarity
  A mathematical metric used to determine how similar two vectors are by measuring the angle between them. In semantic search, a cosine similarity near 1.0 means two text chunks have almost identical meanings, regardless of the specific vocabulary used.

Data Poisoning
  A training-phase attack where an adversary infiltrates a data pipeline and subtly alters the training examples or labels, causing the AI to learn a malicious, hidden rule (a "backdoor") that the adversary can exploit later in combat.

Decoder
  The second half of an Autoencoder that attempts to decompress the latent space bottleneck back into the original data format.

Decision Tree
  A supervised learning model that classifies data by recursively splitting it into branches based on the feature that best purifies the resulting groups (often measured by {term}`Gini Impurity`).

Deep Learning
  A subset of machine learning utilizing neural networks with many hidden layers (deep architectures) capable of autonomously engineering complex, hierarchical features from raw, unstructured data (like images or RF signals).

Document Chunking
  The strategy of splitting massive text documents into smaller, overlapping segments (chunks) so they can be individually embedded, searched, and fit within an LLM's limited context window.

Dropout
  A regularization technique where a random percentage of neurons in a layer are temporarily disabled during each training step. This prevents the network from relying on specific neurons to memorize noise, forcing it to learn robust, generalized features.

Early Stopping
  A regularization technique that monitors the validation loss during training. It automatically halts the training process when the model stops improving on unseen data, saving the optimal weights before overfitting occurs.

Edge Device
  A lightweight, low-power hardware payload (like a drone's onboard computer) capable of processing neural network inference locally without requiring a connection to a centralized cloud server.

Encoder
  The first half of an Autoencoder that compresses raw input data down into a mathematically dense bottleneck.

Epoch
  One complete pass of the entire training dataset through the neural network during the training phase.

Expert System
  An early form of Symbolic AI that attempts to mimic human expertise by utilizing an inference engine to process a massive knowledge base of hardcoded, explicit "if-then" rules.

Explainable AI (XAI)
  A field of research and a set of mathematical tools (like SHAP) designed to make the internal mechanics and outputs of "black box" machine learning models understandable to human operators, ensuring trust and traceability.

F1-Score
  The harmonic mean of Precision and Recall. It provides a single, balanced mathematical metric to evaluate a model's performance, especially when dealing with highly imbalanced datasets.

False Negative (Type II Error)
  A failure where the model predicts that a condition is absent, but it is actually present (e.g., the AI predicts "no threat," but an enemy is there). In warfare, this is often the most fatal type of error.

False Positive (Type I Error)
  A failure where the model predicts that a condition is present, but it is actually absent (e.g., the AI predicts an enemy, but it is a civilian). This leads to false alarms, wasted resources, and collateral damage.

Feature Importance
  A technique used to interpret machine learning models by calculating and ranking exactly which input variables (features) had the greatest mathematical impact on the model's final predictions.

Feature Map
  The output image or matrix produced after a convolution filter has scanned across the input data, mathematically highlighting exactly where specific visual features (like lines or shapes) are located.

Feedforward Architecture
  A neural network design where information moves in only one direction—from the input layer, through hidden layers, directly to the output layer—without looping back.

Filter (Kernel)
  A small grid of learnable weights used in a CNN that slides across an image to detect specific patterns, such as horizontal lines or color gradients.

Fine-Tuning
  An advanced form of transfer learning where the upper layers of a pre-trained neural network are "unfrozen" and trained on domain-specific data to adapt the model to new geometric realities (like shifting from ground-level to top-down satellite imagery).

Forward Propagation
  The process of pushing raw input data forward through the hidden layers of a neural network to generate a final prediction. It involves calculating the weighted sums and applying activation functions at each neuron in the sequence.

Generalization
  The ultimate goal of a machine learning model: the ability to learn the true underlying patterns of a dataset so successfully that it performs highly accurately on brand new, unseen data. 

Gini Impurity
  A mathematical metric used by Decision Trees to measure how "mixed" or impure a node of data is. A score of 0 means perfect purity (the node contains only one class).

Gradient
  A mathematical vector of partial derivatives representing the direction of steepest ascent for the loss function.

Gradient Descent
  The optimization algorithm used to minimize a network's loss. It updates the network's weights iteratively by taking small mathematical steps in the opposite direction of the {term}`Gradient`.

Grounding
  The process of forcing an LLM to generate its response based strictly on verified, provided context (such as retrieved documents) rather than relying on its generalized, pre-trained memory. 

Hallucinations
  A phenomenon where a Large Language Model (LLM) or generative AI generates false, fabricated, or nonsensical information but presents it with high confidence as if it were a verified fact. This occurs because the model is predicting statistically likely word sequences, not retrieving verified truth.

Hardware Agent (Autonomous Robot)
  An agent equipped with physical sensors (to perceive the environment) and physical actuators (to manipulate the environment).

Hidden Layer
  A layer of artificial neurons situated between the input and output layers of a neural network. These layers are responsible for learning abstract, hidden features in the data.

Hierarchical Feature Learning
  The process by which deep neural networks autonomously learn simple concepts (like lines and edges) in early layers and mathematically combine them into complex tactical concepts (like vehicles or radar structures) in deeper layers.

Hyperplane
  The mathematical decision boundary drawn by a Support Vector Machine to separate classes. In 2D space, it is a line; in 3D space, it is a flat plane.

Inference Latency
  The time it takes for a deployed neural network to process a single input (like a video frame) and output a prediction. Critical for autonomous edge devices tracking fast-moving targets.

Inference Mechanism
  The processing engine or program in a knowledge-based system that parses the declarative rules stored in the Knowledge Base to draw conclusions or answer queries.

Instruct-Tuning
  The secondary training phase for LLMs where the model is fine-tuned on explicitly labeled prompt/response pairs. This transforms a base model into a highly obedient assistant capable of following specific tactical instructions.

Irreducible Error
  The inherent noise, randomness, or "fog of war" in a dataset that no machine learning model can ever predict or eliminate.

Kernel Trick
  A mathematical technique used by Support Vector Machines (SVMs) to project non-linear, complex data into higher dimensions where it can be easily sliced by a flat hyperplane.

Knowledge Base (KB)
  A centralized repository of declarative rules, logic, or data that is explicitly separated from the system's processing/inference mechanism, allowing the knowledge to be updated independently.

Laplace Smoothing (Add-One Smoothing)
  A mathematical technique used in Naive Bayes to prevent the formula from collapsing to zero when the model encounters a feature (like a new vocabulary word) it has never seen before.

Large Language Model (LLM)
  A massive deep learning model based on the Transformer architecture, trained on enormous datasets to understand, generate, and reason over human language and unstructured text.

Latent Space (Bottleneck)
  The highly compressed, mathematical representation of data found in the middle layer connecting an encoder and decoder within an Autoencoder.

Learning Rate
  A hyperparameter ($\alpha$) that dictates how large of a step the network takes during {term}`Gradient Descent`. If it is too large, the network overcorrects erratically; if it is too small, learning is painfully slow.

Likelihood
  In Bayes' Theorem, the probability of observing a specific piece of evidence assuming that a particular class is true.

Logistic Regression
  A foundational classification algorithm that passes a linear equation through a {term}`Sigmoid Function` to predict the probability of a binary outcome.

Long-Term Memory (Agentic)
  An AI agent's ability to store and recall information across different, isolated chat sessions, typically implemented using a Vector Database and Retrieval-Augmented Generation (RAG).

Loss Function (Cost Function)
  A mathematical function that evaluates how far off a network's predictions are from the true targets. The network's entire training goal is to minimize this value.

Max Pooling
  A downsampling operation commonly used in CNNs that slides a window across a feature map, keeping only the most prominent signal (the maximum value) while discarding the rest. This reduces computational load and provides translation invariance.

Mean Squared Error (MSE)
  A common loss function used for regression tasks (predicting continuous numbers) that calculates the average squared difference between the predicted values and the actual target values.

Naive Assumption (Independence Assumption)
  The core (and mathematically flawed) assumption in Naive Bayes that every feature in a dataset is completely independent of every other feature.

Observation (ReAct)
  The step in the ReAct loop where the result of an external tool call is returned and injected back into the agent's context window, allowing the agent to evaluate the outcome of its action.

Overfitting
  A failure state (driven by high variance) where a model becomes so overly complex that it memorizes the random noise in its training data rather than learning the underlying tactical rules. It performs perfectly in training but fails on unseen data.

Padding (Zero-Padding)
  The technique of adding a border of zero-value pixels around the edges of an input image before applying a convolution. This prevents the resulting feature map from shrinking and preserves edge information.

Perceptron
  The simplest historical form of an artificial neuron. It takes multiple inputs, calculates a weighted sum, adds a bias, and passes the result through a hard step-function to output a 1 or a 0.

Positional Encoding
  A mathematical mechanism in Transformers that tags every input token with a unique signature indicating its exact order in the sentence, allowing the model to process all words in parallel without losing sequential context.

Posterior
  In Bayes' Theorem, the final, updated probability of a class being true *after* factoring in the new evidence.

Precision
  An operational metric answering: "Out of all the times the model predicted a positive target, how many were actually correct?" Calculated as TP / (TP + FP).

Precision-Recall (PR) Curve
  A graph that plots Precision against Recall across different probability thresholds. It is highly valuable for evaluating models trained on severely imbalanced datasets, visually demonstrating the tactical tradeoff between catching all threats and avoiding false alarms.

Pre-Training
  The initial, massive training phase of an LLM where it ingests vast amounts of unstructured internet text and plays a continuous game of "predict the next word" to learn grammar, facts, and logic.

Prior
  In Bayes' Theorem, the baseline, historical probability of a class occurring *before* any new evidence is observed.

Random Forest
  A powerful ensemble model that builds hundreds of shallow Decision Trees and averages their predictions to achieve high accuracy while avoiding the overfitting trap of single trees.

ReAct (Reasoning and Acting)
  An agentic architecture and prompting strategy that forces an LLM to alternate between articulating explicit reasoning (Thought) and executing tool calls (Action), using the results (Observation) to guide subsequent steps.

Recall (Sensitivity)
  An operational metric answering: "Out of all the true positive targets that actually existed in the environment, how many did the model successfully find?" Calculated as TP / (TP + FN).

Reconstruction Error
  The mathematical difference (often measured via MSE) between an original input and an Autoencoder's attempted reconstruction of it. Spikes in this error trigger anomaly alerts.

Reflex Agent
  An AI agent that makes decisions purely by reacting to current, immediate input/perceptions, without relying on internal memory of past events.

Regularization
  A set of techniques (like {term}`Dropout` or {term}`Early Stopping`) used to mathematically penalize complex models to prevent them from overfitting the training data.

ReLU (Rectified Linear Unit)
  A highly efficient and popular activation function used in deep learning hidden layers. It outputs the raw input directly if it is positive, and outputs zero if it is negative.

Retrieval-Augmented Generation (RAG)
  An AI architecture that enhances a Large Language Model's responses by first retrieving verified facts from an external database and injecting them into the model's prompt, effectively grounding the AI in proprietary or real-time intelligence.

ROC Curve (Receiver Operating Characteristic)
  A graph showing the performance of a classification model at all classification thresholds. It plots the True Positive Rate against the False Positive Rate. The area under this curve (AUC) provides an aggregate measure of performance across all possible thresholds.

Self-Attention
  The core mechanism of a Transformer that allows the model to look at every single word in a document simultaneously and mathematically weigh how strongly each word relates to every other word, creating deep contextual understanding.

Semantic Search
  A highly advanced database search technique that uses Vector Embeddings to find documents based on the actual *meaning* and context of a user's query, rather than relying on exact keyword matches.

SHAP (SHapley Additive exPlanations)
  A mathematical method used in Explainable AI (XAI) based on cooperative game theory. It assigns a specific numerical value to each input feature, representing exactly how much that feature contributed to pushing the model's final prediction away from the baseline average.

Short-Term Memory (Agentic)
  An AI agent's immediate, temporary memory, strictly limited by the model's context window. It tracks the immediate sequence of thoughts, actions, and observations during the current operational loop.

Sigmoid Function
  A mathematical function that squashes any real number into a valid probability value bounded strictly between 0.0 and 1.0. 

Software Agent
  An AI program that lacks physical actuators and exists entirely digitally, calculating results or taking actions based purely on data inputs.

Stride
  A hyperparameter defining how many pixels a convolution filter or pooling window shifts at a time as it slides across an image or feature map.

Supervised Learning
  A machine learning paradigm where the algorithm is trained on historical data that already includes the correct answers (labels). It learns to map inputs to those known outputs.

Symbolic AI
  An early paradigm of AI that relied exclusively on explicit, hardcoded logic and symbol processing (like `if-then` rules) rather than learning from data.

Testing Set
  A portion of the dataset (usually 20%) that is strictly hidden from the machine learning algorithm during training. It is used as a final "combat simulation" to test how well the model generalizes to unseen data.

Thought (ReAct)
  The step in the ReAct loop where the agent explicitly articulates its internal reasoning, assessing the current situation and determining what action to take next.

Tokenization
  The process of chopping raw text down into sub-word chunks (tokens) and translating them into mathematical ID numbers so they can be processed by a neural network.

Training Set
  The portion of the dataset (usually 80%) exposed to the machine learning algorithm. The model uses this data to adjust its internal math and learn patterns.

Transfer Learning
  A rapid deployment technique where a massive neural network, pre-trained on millions of generic data points, has its base layers "frozen" and its final classification head re-trained for a specific tactical domain.

Transformer
  A breakthrough neural network architecture that relies heavily on self-attention and positional encoding to process sequential data (like text) entirely in parallel, forming the architectural foundation of all modern Large Language Models.

Translation Invariance
  The ability of a neural network (particularly a CNN using pooling) to recognize a target regardless of where it physically shifted or moved within the camera frame.

Underfitting (High Bias)
  A failure state where a model is too rigid or simple to learn the underlying patterns in the training data, resulting in poor predictive performance across the board.

Unsupervised Learning
  A machine learning paradigm where the algorithm is given raw, unlabeled data and must autonomously discover hidden structures, clusters, or patterns on its own.

Variance (Statistical Error)
  The error introduced when a model is overly sensitive to tiny fluctuations or noise in the training data. High variance leads to {term}`Overfitting`.

Vector Database
  A specialized database designed to store and query high-dimensional vector embeddings, forming the backbone of semantic search in a RAG pipeline.

Vector Embeddings
  A dense array of floating-point numbers representing the core semantic meaning of a piece of text (or image), allowing algorithms to measure the "distance" or similarity between distinct concepts mathematically.

Weights
  The learnable parameters ($w_i$) in a neural network that determine the importance or influence of a specific input feature on the final prediction.



```