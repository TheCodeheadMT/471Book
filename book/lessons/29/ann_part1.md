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


>
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


![Aritifical Neural Networks Intro](../../figures/ann1_info.png "ANN Infographfic")