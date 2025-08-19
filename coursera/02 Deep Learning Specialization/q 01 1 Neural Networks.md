## 1 Neural Networks

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

