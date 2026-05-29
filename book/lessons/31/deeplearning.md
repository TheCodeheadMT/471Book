# L31: Deep Learning

In the previous lessons, we built shallow neural networks (1-2 hidden layers) and trained them to solve specific, highly structured problems. However, the modern battlespace does not always provide neatly organized tabular data. We must process raw, high-dimensional data: live video feeds from drones, gigabytes of raw network traffic, and chaotic electromagnetic radio frequencies.


:::{admonition} Lesson Objectives
:class: note
  * Explain hierarchical feature learning
  * Describe modern architectures
  * Identify real-world deep learning applications
:::

Enter **Deep Learning**. By stacking dozens (or hundreds) of hidden layers, a neural network ceases to be a simple mathematical calculator and becomes a **hierarchical feature extractor**. In this lesson, we will explore how deep architectures autonomously learn the underlying structure of reality, and how we can rapidly adapt these massive models for tactical advantage.

<br>

## Hierarchical Feature Learning

In a standard shallow network, the hidden layer tries to map the raw inputs directly to the final prediction. In a Deep Neural Network (DNN), each layer learns increasingly complex, abstract representations of the data.

If we feed raw pixels of a satellite image into a deep network:

* **Layer 1 (Low-Level):** Learns to detect basic edges, lines, and color gradients.

* **Layer 5 (Mid-Level):** Combines edges to detect shapes, wheels, and metallic textures.

* **Layer 50 (High-Level):** Combines shapes to recognize complete tactical objects: "T-72 Tank" or "Supply Truck."

The network mathematically engineers its own features. The human operator no longer has to tell the algorithm *what* to look for; the algorithm figures it out autonomously.

<br>

<hr width="100%" size="4" color="black">


## Transfer Learning

**Scenario:** Cargo Logistics Classification. You need to deploy an AI on a surveillance drone to classify vehicles at a logistics hub as either "Military Transport" or "Civilian Cargo."

Training a deep image classification network from scratch requires millions of labeled images, hundreds of GPUs, and weeks of training time—resources you do not have in a forward operating base.

### The Math: The Transfer Protocol

Instead of starting with randomized weights, we use {term}`Transfer Learning`. We take a massive, pre-trained network (like MobileNet or ResNet) that has already spent weeks learning how to recognize millions of everyday objects (cats, dogs, cars, planes). This network already possesses perfect low-level and mid-level feature detectors.

We mathematically "freeze" the weights of these lower layers so they cannot be altered by {term}`Gradient Descent`. We then slice off the original output layer (the "head") and attach a brand new, randomized output layer designed specifically for our 2 classes.

$$W_{base} = \text{Frozen (No Gradient Updates)}$$

$$W_{head} = \text{Trainable (Updates via Gradient Descent)}$$

We only train the final layer. The network learns to map its pre-existing knowledge of "shapes and wheels" directly to our specific tactical categories in minutes rather than weeks.

<br>

<hr width="100%" size="4" color="black">


## Unsupervised Autoencoders

**Scenario:** Cyber Intrusion Detection. You are defending a secure Air Force network. Hackers do not use the same tactics twice, meaning you cannot train a Supervised classifier on "known" attacks, because tomorrow's attack will look entirely different. We must use an Unsupervised deep learning architecture.

### The Math: The Autoencoder

An {term}`Autoencoder` is a neural network trained to recreate its own input. It consists of an {term}`Encoder` that compresses the data down into a tiny bottleneck (the Latent Space, $Z$), and a {term}`Decoder` that attempts to reconstruct the original data ($X'$) from that bottleneck.

$$Z = \text{Encoder}(X)$$

$$X' = \text{Decoder}(Z)$$

$$\text{Loss} = \text{MSE}(X, X')$$

**The Tactical Application:** We train the Autoencoder *exclusively* on benign, normal network traffic. The network learns to perfectly compress and decompress normal behavior.

When a Zero-Day cyber attack occurs, the anomalous traffic is fed into the network. Because the network has never seen this pattern, it fails to compress it properly, resulting in a massive, mathematically measurable {term}`Reconstruction Error`. If the error spikes above a threshold, we isolate the network node.


<br>

<hr width="100%" size="4" color="black">

## Electronic Warfare (Case Study)

**The Historical Paradigm:** For decades, classifying radar and radio frequency (RF) signals in Electronic Warfare relied on hand-crafted statistical filters. Signal Intelligence (SIGINT) operators would capture a signal, run a Fast Fourier Transform (FFT) to convert it from the time domain to the frequency domain, and manually engineer features (pulse width, frequency modulation, amplitude). These manual features were then fed into a basic classifier (like a Support Vector Machine). If the adversary updated their radar modulation by just a few hertz, the manual filters failed, and human engineers had to rewrite the software.

**The Deep Learning Paradigm:**
Modern EW systems have abandoned manual feature engineering. Today, operators feed raw In-Phase and Quadrature (IQ) radio samples directly into massive deep networks—specifically 1-Dimensional Convolutional Neural Networks (1D-CNNs).

Because these networks are *deep*, they learn the hierarchical features of the RF spectrum autonomously. The lower layers learn to filter basic noise, the middle layers learn pulse structures, and the final layers classify the specific adversary emitter. If the adversary changes their tactics, the Air Force simply collects a new batch of raw IQ data, runs a transfer learning update (like we did in the cargo lab), and pushes the updated weights back to the electronic attack pods on the aircraft in a matter of hours. Deep Learning has shifted EW from hardware-defined combat to software-defined, highly agile warfare.


