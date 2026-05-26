Here is the complete set of content, a laboratory exercise, and practice problems for **L29 Neural Networks I**, formatted to match your previous textbook and lab structures.

---

# L29: Neural Networks I

Welcome to the next leap in the Smart Air Force Warfighter curriculum. While statistical models like Logistic Regression and Support Vector Machines are powerful, they often require humans to manually engineer features. Artificial Neural Networks (ANNs) take inspiration from the human brain, allowing systems to automatically learn complex, non-linear representations of raw data. In this lesson, we will explore the fundamental building block of deep learning: the artificial neuron, and how connecting them creates feedforward architectures capable of solving highly complex tactical problems.

## The Artificial Neuron (The Perceptron)

**Scenario:** UAV Sortie Viability. You need to predict whether an autonomous UAV has enough battery to complete a sortie based on three telemetry features: Payload Weight, Headwind Speed, and Mission Distance.

### The Math: The Weighted Sum

An artificial neuron receives multiple inputs, applies a specific weight to each (representing the importance of that input), and adds a bias term (a baseline threshold).

The weighted sum $z$ of a single neuron is calculated as:

$$z = \sum_{i=1}^{n} w_i x_i + b$$

Once the weighted sum is calculated, it is passed through an **activation function** to determine the neuron's final output. For binary classification (like Success or Failure), we often use the **Sigmoid** activation function to squash the output into a probability between **0.0** and **1.0**.

> **Example 29.1 - The Single Neuron**
> Let's predict UAV battery viability. Our trained artificial neuron has the following weights:
> * Payload ($x_1$): $w_1 = -0.2$
> * Headwind ($x_2$): $w_2 = -0.5$
> * Distance ($x_3$): $w_3 = -0.1$
> * Bias: $b = 40$
> 
> 
> We receive a mission profile: **50 kg** payload, **20 knots** headwind, **150 km** distance.
> 1. Calculate the weighted sum: $z = (-0.2 \times 50) + (-0.5 \times 20) + (-0.1 \times 150) + 40$
> 2. $z = -10 - 10 - 15 + 40 = 5$
> 3. Apply the Sigmoid function: $P = \frac{1}{1 + e^{-5}}$
> 4. Since $e^{-5} \approx 0.0067$, the probability is $P = \frac{1}{1.0067} \approx 0.993$
> 
> 
> **Result:** The neuron predicts a **99.3%** probability that the UAV will successfully complete the sortie. The mission is cleared.

---

## Feedforward Architectures & Non-Linearity

**Scenario:** Sonar Classification. You must classify raw sonar returns as either Subsurface Mines or Rocks.

A single neuron can only draw linear decision boundaries. To classify complex, non-linear sonar data, we must stack neurons into **Hidden Layers**. Data moves in one direction—from inputs, through hidden layers, to the output. This is a **Feedforward Architecture**.

### The Math: Activation Functions

If we only use linear combinations, stacking layers is mathematically useless (a linear function of a linear function is still linear). We must introduce non-linearity using activation functions in the hidden layers:

* **ReLU (Rectified Linear Unit):** $f(x) = \max(0, x)$. It outputs the input directly if positive, otherwise, it outputs zero. This simple function allows networks to model highly complex curves and is computationally efficient.
* **Sigmoid:** $\sigma(x) = \frac{1}{1 + e^{-x}}$. Typically reserved for the final output node to yield a probability.

> **Example 29.2 - Two-Layer Forward Propagation**
> Imagine a simplified sonar system with 2 input features, a hidden layer of 2 neurons using ReLU, and 1 output neuron using Sigmoid.
> Input vector $X = [2, -1]$.
> **Hidden Layer (ReLU):**
> * Neuron 1 Weights: $w = [1, 1]$, Bias = $0$.
> * $z_1 = (2 \times 1) + (-1 \times 1) + 0 = 1$.
> * Activation: $\max(0, 1) = 1$.
> 
> 
> * Neuron 2 Weights: $w = [-1, 2]$, Bias = $1$.
> * $z_2 = (2 \times -1) + (-1 \times 2) + 1 = -3$.
> * Activation: $\max(0, -3) = 0$.
> 
> 
> 
> 
> **Output Layer (Sigmoid):**
> * Output Weights: $w = [2, -2]$, Bias = $-1$.
> * Inputs to this layer are the activations from the hidden layer: $[1, 0]$.
> * $z_{out} = (1 \times 2) + (0 \times -2) - 1 = 1$.
> * Activation: $\frac{1}{1 + e^{-1}} \approx 0.73$.
> 
> 
> 
> 
> **Result:** The network predicts a **73%** probability that the sonar return is a Subsurface Mine.

---

## Complex Architectures: Multiple Outputs

**Scenario:** Strike Package Optimization. You must simultaneously predict the estimated fuel consumption and the projected time-on-target for a strike package.

Unlike statistical models that usually output a single prediction, neural networks can be designed to output multiple predictions at once. A **Multi-Output Feedforward Network** takes a single input vector, passes it through shared hidden layers (which extract universal features like weather conditions and aircraft drag), and then branches off into distinct output nodes.

One output node might use a linear activation to predict continuous fuel consumption in gallons, while another output node predicts the flight time in minutes. By sharing hidden layers, the network learns a more robust internal representation of the battlespace, improving the accuracy of both predictions.

---

## L29 Lab: Forward Propagation from Scratch

```python
# EX: Forward Propagation (Mines vs. Rocks Sonar Returns)
# In this lab, we will execute forward propagation manually using NumPy to understand the underlying matrix math.

import numpy as np

# 1. Define the Activation Functions
def relu(z):
    return np.maximum(0, z)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 2. Simulate Sonar Input Data (1 Sample, 3 Features: Frequency, Ping Duration, Amplitude)
X = np.array([[0.8, 1.2, -0.5]])

# 3. Define the Network Architecture (3 Inputs -> 4 Hidden Neurons -> 1 Output)
# Weights and biases for Hidden Layer 1
W1 = np.array([
    [ 0.2, -0.1,  0.4,  0.8],
    [ 0.5,  0.6, -0.3, -0.2],
    [-0.4,  0.3,  0.1,  0.5]
])
b1 = np.array([[0.1, -0.2, 0.0, 0.1]])

# Weights and biases for Output Layer
W2 = np.array([
    [ 0.6],
    [-0.5],
    [ 0.8],
    [-0.3]
])
b2 = np.array([[-0.1]])

# 4. Execute Forward Propagation
# Hidden Layer: Weighted sum (Z1) then ReLU (A1)
Z1 = np.dot(X, W1) + b1
A1 = relu(Z1)

# Output Layer: Weighted sum (Z2) then Sigmoid (A2)
Z2 = np.dot(A1, W2) + b2
A2 = sigmoid(Z2)

print("--- Forward Propagation Results ---")
print(f"Input Features (X): {X}")
print(f"Hidden Layer Pre-Activation (Z1): {Z1}")
print(f"Hidden Layer Post-ReLU (A1): {A1}")
print(f"Output Pre-Activation (Z2): {Z2}")
print(f"Final Prediction Probability (A2): {A2[0][0]:.4f}")

# Threshold logic
if A2[0][0] > 0.5:
    print("Classification: Subsurface Mine Detected")
else:
    print("Classification: Harmless Rock")

```

---

## L29 Practice Problems

**1. Perceptron Weighted Sum**
A single artificial neuron is used to detect radar spoofing. It uses two features: Signal-to-Noise Ratio (SNR) and Frequency Shift. The weights are $w_1 = 3.0$ and $w_2 = -2.0$, with a bias of $b = -1.0$. A suspicious return has an SNR of $1.5$ and a Frequency Shift of $1.0$. Calculate the weighted sum $z$.

**2. Activation Functions: ReLU**
Using the weighted sum $z = -2.5$ from a hidden layer neuron analyzing satellite imagery, what is the output of this neuron if it uses a Rectified Linear Unit (ReLU) activation function?

**3. Two-Layer Forward Propagation**
A simplified neural network detects cyber intrusions. Input $X = [1, 2]$. The hidden layer has two neurons with weights $W_1 = [ [1, -1], [0, 1] ]$ and biases $b_1 = [0, -1]$. The hidden layer uses a ReLU activation function. The output layer has weights $W_2 = [1, 1]$ and bias $b_2 = 0$. Calculate the pre-activation output $z_{out}$ of the final layer.

**4. Multi-Output Network Parameters**
You are designing a neural network that takes 5 inputs (weather conditions). It feeds into a single hidden layer of 10 neurons. The network branches into two separate outputs: one predicting fuel consumption, the other predicting time-on-target. Assuming every layer is fully connected and includes bias terms, how many total learnable parameters (weights + biases) does this network have?


