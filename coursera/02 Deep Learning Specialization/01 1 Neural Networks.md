## 1 Neural Networks



### Key Points



#### 1. ⚡ AI and Deep Learning Impact  
- AI is considered the new electricity due to its transformative potential across industries.  
- Deep learning is a subset of AI that enables learning from data without explicit programming.

#### 2. 🧠 Neural Networks Basics  
- A neural network consists of layers of interconnected neurons that process inputs to produce outputs.  
- Neural networks can handle both structured data (e.g., house features) and unstructured data (e.g., images, audio).  
- Supervised learning trains neural networks using input-output pairs.

#### 3. 📊 Neural Network Applications  
- Neural networks are used in housing price prediction, online advertising, image recognition, speech recognition, machine translation, and autonomous driving.  
- Types of neural networks include Standard NN, Convolutional NN (for images), and Recurrent NN (for sequences).

#### 4. 🚀 Reasons for Deep Learning Success  
- Deep learning progress is driven by the availability of large datasets, increased computational power (GPUs), and improved algorithms.

#### 5. 🧮 Logistic Regression Fundamentals  
- Logistic regression is a binary classification model that outputs probabilities using the sigmoid function.  
- The logistic regression cost function measures prediction error and is minimized during training.  
- Gradient descent updates model parameters by moving in the direction that reduces the cost function.

#### 6. 🔄 Vectorization  
- Vectorization uses matrix and vector operations to process multiple training examples simultaneously, improving computational efficiency.  
- Broadcasting in Python allows arithmetic operations between arrays of different shapes by automatically expanding dimensions.

#### 7. 🕸️ One Hidden Layer Neural Networks  
- A hidden layer applies weighted sums and non-linear activation functions to inputs before passing to the output layer.  
- Non-linear activation functions (sigmoid, tanh, ReLU, Leaky ReLU) enable neural networks to learn complex patterns.  
- Forward propagation computes outputs layer-by-layer; backpropagation computes gradients for training.

#### 8. 🎲 Initialization and Deep Networks  
- Random initialization of weights is necessary to break symmetry; zero initialization causes neurons to learn the same features.  
- Deep neural networks have multiple hidden layers, allowing hierarchical feature learning.  
- Deep networks can represent some functions exponentially more efficiently than shallow networks.

#### 9. ⚙️ Parameters vs Hyperparameters  
- Parameters (weights and biases) are learned during training.  
- Hyperparameters (learning rate, number of layers, batch size, etc.) are set before training and control the learning process.

#### 10. 🧠 Neural Networks and the Brain  
- Neural networks are loosely inspired by the brain’s structure but are much simpler computational models.  
- Applied deep learning involves iterative cycles of idea, experiment, and code.



<br>

## Study Notes



### Study Notes: Introduction to Neural Networks and Deep Learning 🤖


### 1. ⚡ The Big Picture: Why AI and Deep Learning Matter

Artificial Intelligence (AI) is often compared to electricity in terms of its transformative power. Just as electricity revolutionized industries like transportation, manufacturing, healthcare, and communications, AI is now poised to bring about a similarly profound change across many fields.

Deep learning, a subset of AI, is at the heart of this transformation. It enables computers to learn from data and make decisions or predictions without being explicitly programmed for every task. This course sequence introduces you to the foundations of deep learning, starting with neural networks, and then moving on to more advanced topics like hyperparameter tuning, convolutional neural networks, and natural language processing.


### 2. 🧠 What is a Neural Network?

At its core, a neural network is a computational model inspired by the human brain. It consists of layers of interconnected nodes (called neurons) that process input data to produce an output. Neural networks are especially powerful for tasks where traditional programming struggles, such as recognizing images, understanding speech, or predicting complex patterns.

#### Example: Housing Price Prediction

Imagine you want to predict the price of a house. The input features might include:

- Size of the house (in square feet)
- Number of bedrooms
- Zip code (location)
- Other factors like wealth or neighborhood quality

A neural network can take these inputs and learn to predict the house price by finding patterns in historical data.

#### Supervised Learning with Neural Networks

Neural networks often learn through supervised learning, where the model is trained on input-output pairs. For example:

- Input: Features of a house (size, bedrooms, zip code)
- Output: Price of the house

The network adjusts its internal parameters to minimize the difference between its predicted output and the actual price.


### 3. 📊 Applications of Neural Networks

Neural networks are versatile and can handle both structured and unstructured data:

- **Structured data:** Tabular data like user age, ad ID, or house features.
- **Unstructured data:** Images, audio, text, and other complex data types.

Some common applications include:

- Real estate price prediction
- Online advertising (predicting if a user will click an ad)
- Image recognition (tagging photos)
- Speech recognition
- Machine translation (translating languages)
- Autonomous driving (understanding the environment)

There are different types of neural networks tailored for specific data types:

- **Standard Neural Networks:** For general tasks with structured data.
- **Convolutional Neural Networks (CNNs):** Specialized for image data.
- **Recurrent Neural Networks (RNNs):** Designed for sequential data like text or audio.


### 4. 🚀 Why is Deep Learning Taking Off Now?

Deep learning has become extremely popular and effective recently due to three main factors:

1. **Data:** The availability of massive datasets allows neural networks to learn complex patterns.
2. **Computation:** Advances in hardware, especially GPUs, enable fast training of large models.
3. **Algorithms:** Improved techniques and architectures make training deep networks more feasible and effective.

Together, these factors have driven rapid progress in AI capabilities.


### 5. 🧮 Basics of Neural Network Programming

#### Logistic Regression: The Starting Point

Before diving into neural networks, it’s important to understand logistic regression, a simple model used for binary classification (e.g., cat vs. non-cat images).

- **Input:** Features (e.g., pixel values of an image)
- **Output:** Probability that the input belongs to a certain class (e.g., cat = 1, non-cat = 0)

The logistic regression model uses a function called the **sigmoid** to map any input to a value between 0 and 1, representing a probability.

#### Cost Function

To train the model, we define a **cost function** (also called a loss function) that measures how far the model’s predictions are from the true labels. For logistic regression, the cost function is:

\[
J(w, b) = -\frac{1}{m} \sum_{i=1}^m \left[ y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) \right]
\]

where:

- \(m\) = number of training examples
- \(y^{(i)}\) = true label for example \(i\)
- \(\hat{y}^{(i)}\) = predicted probability for example \(i\)

The goal is to find parameters \(w\) and \(b\) that minimize this cost.

#### Gradient Descent

To minimize the cost function, we use **gradient descent**, an iterative optimization algorithm. It updates the parameters in the direction that reduces the cost:

\[
w := w - \alpha \frac{\partial J}{\partial w}
\]
\[
b := b - \alpha \frac{\partial J}{\partial b}
\]

where \(\alpha\) is the learning rate, controlling the step size.

#### Derivatives and Computation Graphs

Understanding derivatives is crucial because gradient descent relies on computing gradients (derivatives of the cost function with respect to parameters). A **computation graph** is a way to visualize and compute these derivatives efficiently by breaking down complex functions into simpler operations.


### 6. 🔄 Vectorization and Efficient Computation

When working with large datasets, using loops to process each example individually is inefficient. Instead, **vectorization** leverages matrix and vector operations to perform computations on all examples simultaneously.

For example, instead of computing the sigmoid for each input one by one, vectorization applies the sigmoid function to an entire matrix of inputs at once. This speeds up training significantly.

Python libraries like NumPy support vectorized operations and **broadcasting**, which automatically expands smaller arrays to match the shape of larger ones during arithmetic operations.


### 7. 🕸️ One Hidden Layer Neural Networks

#### What is a Hidden Layer?

A neural network with one hidden layer has three layers:

- **Input layer:** Receives the input features.
- **Hidden layer:** Processes inputs through neurons with activation functions.
- **Output layer:** Produces the final prediction.

Each neuron in the hidden layer computes a weighted sum of inputs, adds a bias, and applies a non-linear **activation function**.

#### Why Non-linear Activation Functions?

Without non-linear activation functions, the network would behave like a linear model regardless of the number of layers. Non-linear functions allow the network to learn complex patterns.

Common activation functions include:

- **Sigmoid:** Outputs values between 0 and 1, useful for probabilities.
- **Tanh:** Outputs values between -1 and 1, centered around zero.
- **ReLU (Rectified Linear Unit):** Outputs zero if input is negative, otherwise outputs input directly. It helps with faster training and avoids some problems of sigmoid/tanh.
- **Leaky ReLU:** A variant of ReLU that allows a small gradient when input is negative, preventing "dead neurons."

#### Forward Propagation

Forward propagation is the process of computing the output of the network given an input by passing data through each layer.

#### Backpropagation and Gradient Descent

To train the network, we use **backpropagation** to compute gradients of the cost function with respect to each parameter. This involves applying the chain rule of calculus through the layers. Then, gradient descent updates the parameters to reduce the error.


### 8. 🎲 Initialization and Deep Neural Networks

#### Random Initialization

Initializing weights randomly is important. If all weights start at zero, neurons learn the same features and the network fails to learn effectively. Random initialization breaks symmetry and allows different neurons to learn different features.

#### What is a Deep Neural Network?

A deep neural network has multiple hidden layers (more than one). These layers allow the network to learn hierarchical features — simple features in early layers and complex features in deeper layers.

For example:

- Logistic regression = 0 hidden layers
- One hidden layer network = 1 hidden layer
- Deep network = 2 or more hidden layers

#### Why Deep Representations?

Deep networks can represent complex functions more efficiently than shallow networks. Some functions require exponentially more neurons in a shallow network compared to a deep one.

#### Forward and Backward Propagation in Deep Networks

The same principles of forward and backward propagation apply, but computations are repeated across many layers. Careful management of matrix dimensions and parameters is essential.


### 9. ⚙️ Parameters vs Hyperparameters

- **Parameters:** These are learned during training (weights and biases).
- **Hyperparameters:** These are set before training and control the learning process, such as learning rate, number of layers, number of neurons per layer, batch size, and number of iterations.

Tuning hyperparameters is an empirical process involving experimentation and iteration.


### 10. 🧠 Connection to the Brain and Final Thoughts

Neural networks are loosely inspired by the brain’s structure, but they are much simpler. The brain’s neurons and synapses are far more complex, but the analogy helps us understand how networks can learn from data.

Applied deep learning is a cycle of:

- **Idea:** Formulate a hypothesis or model.
- **Experiment:** Train and test the model.
- **Code:** Implement and optimize the model.

This iterative process drives progress in AI.


### Summary

This introductory lecture covers the foundations of neural networks and deep learning, starting from basic concepts like logistic regression and gradient descent, moving through vectorization and activation functions, and culminating in the structure and training of deep neural networks. Understanding these basics is essential for exploring more advanced topics like convolutional networks and sequence models.



<br>

## Questions



#### 1. What are the main reasons deep learning has recently become so successful?  
A) Availability of large datasets  
B) Advances in hardware like GPUs  
C) The invention of new programming languages  
D) Improved algorithms and architectures  

#### 2. Which of the following are true about supervised learning with neural networks?  
A) The model learns from input-output pairs  
B) It requires labeled data for training  
C) It can only be used for classification tasks  
D) The network adjusts parameters to minimize prediction error  

#### 3. Why are non-linear activation functions necessary in neural networks?  
A) To allow the network to learn complex, non-linear patterns  
B) To ensure the network behaves like a linear model  
C) To enable multiple layers to add representational power  
D) To prevent the network from overfitting  

#### 4. Which of the following are common activation functions used in neural networks?  
A) Sigmoid  
B) Tanh  
C) ReLU  
D) Linear  

#### 5. What happens if all weights in a neural network are initialized to zero?  
A) The network will learn different features in each neuron  
B) All neurons will learn the same features, limiting learning  
C) The network will fail to break symmetry  
D) The network will converge faster  

#### 6. Which statements about gradient descent are correct?  
A) It updates parameters to minimize the cost function  
B) It requires computing derivatives of the cost function  
C) It always finds the global minimum of the cost function  
D) The learning rate controls the size of parameter updates  

#### 7. What is the purpose of vectorization in neural network programming?  
A) To process multiple training examples simultaneously  
B) To avoid explicit for-loops for efficiency  
C) To reduce the number of parameters in the model  
D) To leverage optimized matrix operations in libraries like NumPy  

#### 8. Which of the following are examples of unstructured data that neural networks can handle?  
A) Images  
B) Tabular data with user age and income  
C) Audio recordings  
D) Text documents  

#### 9. In the context of neural networks, what is a “hidden layer”?  
A) The input layer that receives raw data  
B) A layer between input and output that processes data  
C) The output layer that produces predictions  
D) A layer that applies activation functions to weighted sums  

#### 10. Which of the following statements about deep neural networks are true?  
A) They have multiple hidden layers  
B) They can represent some functions more efficiently than shallow networks  
C) They always require exponentially more neurons than shallow networks  
D) Forward and backward propagation are repeated across all layers  

#### 11. What is the role of the cost (loss) function in training neural networks?  
A) To measure how well the model’s predictions match the true labels  
B) To initialize the weights of the network  
C) To guide the optimization process during training  
D) To determine the network architecture  

#### 12. Which of the following are true about the sigmoid activation function?  
A) It outputs values between 0 and 1  
B) It is centered around zero  
C) It can cause vanishing gradient problems in deep networks  
D) It is the best choice for hidden layers in deep networks  

#### 13. What is backpropagation used for in neural networks?  
A) To compute gradients of the cost function with respect to parameters  
B) To perform forward propagation of inputs through the network  
C) To update weights using gradient descent  
D) To initialize the network parameters randomly  

#### 14. Which of the following are hyperparameters in deep learning?  
A) Learning rate  
B) Number of hidden layers  
C) Weights and biases  
D) Batch size  

#### 15. Why is the ReLU activation function often preferred over sigmoid or tanh in deep networks?  
A) It helps avoid vanishing gradients  
B) It outputs values between -1 and 1  
C) It allows faster training convergence  
D) It is linear for all input values  

#### 16. Which of the following best describe the relationship between parameters and hyperparameters?  
A) Parameters are learned during training  
B) Hyperparameters are set before training and control learning  
C) Parameters include learning rate and batch size  
D) Hyperparameters include weights and biases  

#### 17. What is the main advantage of deep representations in neural networks?  
A) They allow learning hierarchical features from simple to complex  
B) They guarantee better performance on all tasks  
C) They reduce the need for large datasets  
D) They can compute some functions with fewer neurons than shallow networks  

#### 18. Which of the following statements about logistic regression are correct?  
A) It is a linear model used for binary classification  
B) It uses the sigmoid function to output probabilities  
C) It can model complex non-linear relationships without hidden layers  
D) Its cost function is based on cross-entropy loss  

#### 19. What is broadcasting in the context of vectorized operations?  
A) Expanding smaller arrays to match the shape of larger arrays during arithmetic  
B) Sending data from one neuron to all neurons in the next layer  
C) Applying the same operation to each element of a vector individually  
D) A method to initialize weights in neural networks  

#### 20. Which of the following are challenges or considerations when training deep neural networks?  
A) Choosing appropriate hyperparameters  
B) Avoiding overfitting through regularization  
C) Ensuring all weights are initialized to zero  
D) Managing computational resources and training time  



<br>

## Answers



#### 1. What are the main reasons deep learning has recently become so successful?  
A) ✓ Availability of large datasets — More data enables better learning.  
B) ✓ Advances in hardware like GPUs — Faster computation allows training large models.  
C) ✗ The invention of new programming languages — Not a key factor in deep learning’s rise.  
D) ✓ Improved algorithms and architectures — Better methods improve training and performance.  

**Correct:** A, B, D


#### 2. Which of the following are true about supervised learning with neural networks?  
A) ✓ The model learns from input-output pairs — Supervised learning requires labeled data.  
B) ✓ It requires labeled data for training — Labels guide the learning process.  
C) ✗ It can only be used for classification tasks — Also used for regression and other tasks.  
D) ✓ The network adjusts parameters to minimize prediction error — Core of training.  

**Correct:** A, B, D


#### 3. Why are non-linear activation functions necessary in neural networks?  
A) ✓ To allow the network to learn complex, non-linear patterns — Without non-linearity, network is linear.  
B) ✗ To ensure the network behaves like a linear model — Opposite of the purpose.  
C) ✓ To enable multiple layers to add representational power — Non-linearity allows depth to matter.  
D) ✗ To prevent the network from overfitting — Activation functions don’t directly prevent overfitting.  

**Correct:** A, C


#### 4. Which of the following are common activation functions used in neural networks?  
A) ✓ Sigmoid — Classic activation for probabilities.  
B) ✓ Tanh — Zero-centered non-linearity.  
C) ✓ ReLU — Popular for deep networks.  
D) ✗ Linear — Linear functions don’t add non-linearity, so rarely used as activation.  

**Correct:** A, B, C


#### 5. What happens if all weights in a neural network are initialized to zero?  
A) ✗ The network will learn different features in each neuron — Symmetry broken only by random init.  
B) ✓ All neurons will learn the same features, limiting learning — Zero init causes identical gradients.  
C) ✓ The network will fail to break symmetry — Prevents diverse feature learning.  
D) ✗ The network will converge faster — Usually slows or stops learning.  

**Correct:** B, C


#### 6. Which statements about gradient descent are correct?  
A) ✓ It updates parameters to minimize the cost function — Core purpose of gradient descent.  
B) ✓ It requires computing derivatives of the cost function — Gradients guide updates.  
C) ✗ It always finds the global minimum of the cost function — Often stuck in local minima or saddle points.  
D) ✓ The learning rate controls the size of parameter updates — Step size in parameter space.  

**Correct:** A, B, D


#### 7. What is the purpose of vectorization in neural network programming?  
A) ✓ To process multiple training examples simultaneously — Speeds up computation.  
B) ✓ To avoid explicit for-loops for efficiency — Loops are slow in Python.  
C) ✗ To reduce the number of parameters in the model — Vectorization is about computation, not model size.  
D) ✓ To leverage optimized matrix operations in libraries like NumPy — Efficient numerical computation.  

**Correct:** A, B, D


#### 8. Which of the following are examples of unstructured data that neural networks can handle?  
A) ✓ Images — Classic unstructured data type.  
B) ✗ Tabular data with user age and income — Structured data.  
C) ✓ Audio recordings — Sequential unstructured data.  
D) ✓ Text documents — Sequential unstructured data.  

**Correct:** A, C, D


#### 9. In the context of neural networks, what is a “hidden layer”?  
A) ✗ The input layer that receives raw data — Input layer is not hidden.  
B) ✓ A layer between input and output that processes data — Hidden layers transform inputs.  
C) ✗ The output layer that produces predictions — Output layer is not hidden.  
D) ✓ A layer that applies activation functions to weighted sums — Activation happens in hidden layers.  

**Correct:** B, D


#### 10. Which of the following statements about deep neural networks are true?  
A) ✓ They have multiple hidden layers — Definition of deep networks.  
B) ✓ They can represent some functions more efficiently than shallow networks — Depth adds representational power.  
C) ✗ They always require exponentially more neurons than shallow networks — Often fewer neurons needed.  
D) ✓ Forward and backward propagation are repeated across all layers — Training involves all layers.  

**Correct:** A, B, D


#### 11. What is the role of the cost (loss) function in training neural networks?  
A) ✓ To measure how well the model’s predictions match the true labels — Quantifies error.  
B) ✗ To initialize the weights of the network — Initialization is separate.  
C) ✓ To guide the optimization process during training — Minimizing cost drives learning.  
D) ✗ To determine the network architecture — Architecture is designed beforehand.  

**Correct:** A, C


#### 12. Which of the following are true about the sigmoid activation function?  
A) ✓ It outputs values between 0 and 1 — Maps inputs to probabilities.  
B) ✗ It is centered around zero — Sigmoid outputs between 0 and 1, not zero-centered.  
C) ✓ It can cause vanishing gradient problems in deep networks — Saturation leads to small gradients.  
D) ✗ It is the best choice for hidden layers in deep networks — ReLU usually preferred.  

**Correct:** A, C


#### 13. What is backpropagation used for in neural networks?  
A) ✓ To compute gradients of the cost function with respect to parameters — Core of training.  
B) ✗ To perform forward propagation of inputs through the network — Forward pass is separate.  
C) ✓ To update weights using gradient descent — Gradients enable updates.  
D) ✗ To initialize the network parameters randomly — Initialization is separate.  

**Correct:** A, C


#### 14. Which of the following are hyperparameters in deep learning?  
A) ✓ Learning rate — Controls training speed.  
B) ✓ Number of hidden layers — Defines network depth.  
C) ✗ Weights and biases — These are parameters learned during training.  
D) ✓ Batch size — Controls number of examples per training step.  

**Correct:** A, B, D


#### 15. Why is the ReLU activation function often preferred over sigmoid or tanh in deep networks?  
A) ✓ It helps avoid vanishing gradients — ReLU gradients don’t saturate for positive inputs.  
B) ✗ It outputs values between -1 and 1 — ReLU outputs zero or positive values only.  
C) ✓ It allows faster training convergence — Simpler gradient and sparse activation help.  
D) ✗ It is linear for all input values — ReLU is piecewise linear, zero for negatives.  

**Correct:** A, C


#### 16. Which of the following best describe the relationship between parameters and hyperparameters?  
A) ✓ Parameters are learned during training — Weights and biases are adjusted.  
B) ✓ Hyperparameters are set before training and control learning — Examples include learning rate.  
C) ✗ Parameters include learning rate and batch size — These are hyperparameters.  
D) ✗ Hyperparameters include weights and biases — These are parameters.  

**Correct:** A, B


#### 17. What is the main advantage of deep representations in neural networks?  
A) ✓ They allow learning hierarchical features from simple to complex — Enables abstraction.  
B) ✗ They guarantee better performance on all tasks — Not always true; depends on data and task.  
C) ✗ They reduce the need for large datasets — Deep networks often require more data.  
D) ✓ They can compute some functions with fewer neurons than shallow networks — Depth can reduce complexity.  

**Correct:** A, D


#### 18. Which of the following statements about logistic regression are correct?  
A) ✓ It is a linear model used for binary classification — Logistic regression is linear in parameters.  
B) ✓ It uses the sigmoid function to output probabilities — Sigmoid maps linear output to [0,1].  
C) ✗ It can model complex non-linear relationships without hidden layers — It’s limited to linear decision boundaries.  
D) ✓ Its cost function is based on cross-entropy loss — Cross-entropy is standard for classification.  

**Correct:** A, B, D


#### 19. What is broadcasting in the context of vectorized operations?  
A) ✓ Expanding smaller arrays to match the shape of larger arrays during arithmetic — Enables element-wise operations.  
B) ✗ Sending data from one neuron to all neurons in the next layer — This is connectivity, not broadcasting.  
C) ✗ Applying the same operation to each element of a vector individually — This is element-wise operation, not broadcasting.  
D) ✗ A method to initialize weights in neural networks — Initialization is unrelated.  

**Correct:** A


#### 20. Which of the following are challenges or considerations when training deep neural networks?  
A) ✓ Choosing appropriate hyperparameters — Critical for good performance.  
B) ✓ Avoiding overfitting through regularization — Prevents poor generalization.  
C) ✗ Ensuring all weights are initialized to zero — Zero init harms learning.  
D) ✓ Managing computational resources and training time — Deep networks are resource-intensive.  

**Correct:** A, B, D

