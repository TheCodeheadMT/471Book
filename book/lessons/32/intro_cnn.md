# L32: Convolutional Neural Networks

In the previous lessons, our neural networks analyzed data by looking at individual features (like airspeed or altitude). If we fed an image into a standard dense network, we would have to flatten the 2D image into a single 1D line of pixels. This mathematically destroys all spatial relationships—the network would forget which pixels were next to each other.

:::{admonition} Lesson Objectives
:class: note

* Explain convolution, stride, padding, and pooling
* Describe how CNNs achieve spatial hierarchy and translation invariance
* Interpret learned visual feature maps
:::

To process visual Intelligence, Surveillance, and Reconnaissance (ISR) feeds, we need an architecture that understands spatial geometry. **Convolutional Neural Networks (CNNs)** solve this by sliding small, mathematical filters across an image, allowing the AI to "see" shapes, edges, and textures exactly how they appear in the battlespace.

<br>

## The Convolution Operation

**Scenario:** Runway Edge Detection. You are analyzing overhead satellite imagery to locate an adversary's hidden airstrip. Before an AI can classify the airstrip, it must first be able to see the edges of the concrete.

### The Math: Convolution, Stride, and Padding

A **Convolution** is a mathematical operation that slides a small grid of numbers—called a **Filter** or **Kernel** ($K$)—over the input image ($I$). At every step, it multiplies the filter's numbers by the image's pixel values and sums them up.

$$S(i, j) = (I * K)(i, j) = \sum_{m} \sum_{n} I(i-m, j-n) K(m, n)$$

To control how this filter moves, we adjust two hyperparameters:

* **Stride ($S$):** How many pixels the filter shifts at a time. A stride of 1 moves pixel-by-pixel. A stride of 2 skips pixels, shrinking the output size.
* **Padding ($P$):** Sliding a 3x3 filter over an image means the filter cannot perfectly center on the very edge pixels. This causes the output image to shrink. To prevent this, we add a border of zero-value pixels (Zero-Padding) around the original image.

You can calculate the exact width of the resulting output map ($O$) using the spatial dimension formula:


$$O = \lfloor \frac{W - F + 2P}{S} \rfloor + 1$$


*(Where $W$ is input width, $F$ is filter width, $P$ is padding, and $S$ is stride).*


<br>

<hr width="100%" size="4" color="black">

## Pooling & Translation Invariance

**Scenario:** Mobile Target Tracking. An adversary's mobile SAM site is moving across a vast operational sector. If the CNN only learns what the SAM site looks like when it is perfectly centered in the frame, the model will fail when the vehicle moves to the top-left corner.

### The Math: Max Pooling

To solve this, CNNs use **Pooling Layers**. The most common is **Max Pooling**. Similar to a convolution, a pooling window slides across the image, but instead of multiplying, it simply takes the maximum pixel value in that window and discards the rest.

This achieves two critical tactical goals:

1. **Dimensionality Reduction:** It aggressively shrinks the size of the data, reducing the computational load on the aircraft's processors.

2. **Translation Invariance:** Because it takes the "maximum" signal (the strongest feature) within a region, it doesn't matter if the target shifted slightly left or right—the strongest signal survives the pooling operation.


<br>

<hr width="100%" size="4" color="black">

## Feature Maps & What the Network "Looks" At

**Scenario:** Target Verification. Your commander asks, "How do we know the AI is actually identifying the T-72 tank, and not just memorizing the trees in the background?"

We answer this by visualizing the intermediate hidden layers. By peeling back the layers of a CNN, we can see the literal image representations the network is engineering.

<br>
<hr width="100%" size="4" color="black">

## Summary Infographic
![CNN Intro](../../figures/L27_infographic.png "CNN Introduction")

<br>
<hr width="100%" size="4" color="black">

## Knowledge Check

1. **The Padding Problem:** If a Data Scientist applies a 5x5 convolution filter to a 1024x1024 pixel satellite image with a stride of 1 and *no* padding, the resulting feature map will shrink. Why does this happen, and what specific technique is used to prevent it?

2. **Translation Invariance:** An autonomous drone is tracking an adversary's mobile missile launcher. The launcher is constantly shifting its position within the drone's camera frame. Explain how a `MaxPooling` layer mathematically ensures the CNN can still recognize the target despite these shifts.

3. **Spatial Hierarchy:** A deep CNN has been trained to classify visual intelligence from an MQ-9 Reaper. If you extract and visualize the Feature Maps from Layer 1, how will they visually differ from the Feature Maps extracted from Layer 50?

