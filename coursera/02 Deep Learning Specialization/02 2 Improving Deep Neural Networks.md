## 2 Improving Deep Neural Networks

[Study Notes](#study-notes)

[Questions](#questions)



### Key Points



#### 1. 🧠 Train/Dev/Test Sets  
- Training set is used to train the model; dev set is used to tune hyperparameters; test set is used only for final evaluation.  
- Mismatched train/test distributions (e.g., training on web images, testing on user images) can degrade model performance.  
- Having only a dev set without a test set can be acceptable in some cases but is not best practice.

#### 2. ⚖️ Bias and Variance  
- High bias means the model underfits: both training and dev errors are high.  
- High variance means the model overfits: training error is low but dev error is high.  
- The goal is to achieve low bias and low variance for good generalization.

#### 3. 🛡️ Regularization  
- Regularization reduces overfitting by penalizing model complexity or adding noise during training.  
- L2 regularization adds a penalty proportional to the square of the weights to the loss function.  
- Dropout randomly drops neurons during training and scales activations at test time ("inverted dropout").  
- Dropout forces the network to spread out weights and not rely on any single feature.  
- Data augmentation and early stopping are other common regularization methods.

#### 4. ⚙️ Normalizing Inputs  
- Normalizing inputs to zero mean and unit variance speeds up training and stabilizes gradients.  
- Unnormalized inputs can cause slow or unstable training due to uneven gradient scales.

#### 5. 🚫 Vanishing and Exploding Gradients  
- Vanishing gradients cause very small updates in early layers, slowing or stopping learning.  
- Exploding gradients cause very large updates, leading to unstable training.

#### 6. ✅ Gradient Checking  
- Gradient checking numerically approximates gradients to verify backpropagation correctness.  
- It should only be used for debugging, not during training.  
- Gradient checking does not work with dropout enabled.  
- Run gradient checks at initialization and possibly after some training.

#### 7. 🚀 Optimization Algorithms  
- Batch gradient descent uses the entire training set per update; mini-batch gradient descent uses small batches for faster, more efficient training.  
- Exponentially weighted averages smooth gradients over time to reduce noise.  
- Momentum adds a fraction of the previous update to accelerate gradient descent.  
- RMSprop adapts learning rates by dividing gradients by a running average of recent magnitudes.  
- Adam combines momentum and RMSprop for efficient optimization.  
- Learning rate decay reduces the learning rate over time to improve convergence.

#### 8. 🔧 Hyperparameter Tuning  
- Random search is more efficient than grid search for hyperparameter tuning.  
- Use coarse-to-fine search strategies and appropriate scales (e.g., logarithmic for learning rates).  
- Hyperparameters should be re-evaluated periodically as models or data change.

#### 9. 🧮 Batch Normalization  
- Batch normalization normalizes layer activations per mini-batch to reduce internal covariate shift.  
- It speeds up training and adds noise similar to dropout, providing a slight regularization effect.  
- During training, mean and variance are computed per mini-batch; during testing, running averages are used.

#### 10. 🖥️ Deep Learning Frameworks  
- Popular frameworks include TensorFlow, PyTorch, Keras, Caffe, MXNet, PaddlePaddle, Theano, Torch, CNTK.  
- Framework choice depends on ease of programming, speed, community support, and open-source governance.



<br>

## Study Notes





### 1. 🧠 Setting Up Your Machine Learning Application

When building a deep learning model, the first step is to properly set up your machine learning (ML) application. This involves organizing your data and defining how you will train and evaluate your model.

#### Train, Dev, and Test Sets

- **Training set:** This is the data your model learns from. For example, if you are building a cat classifier, your training set might consist of cat pictures collected from various webpages.
- **Development (dev) set:** This set is used to tune your model’s hyperparameters and make decisions about the model architecture. It should be representative of the data your model will see in real use.
- **Test set:** This is used only at the very end to evaluate the final performance of your model. It should be completely separate from training and dev sets to provide an unbiased estimate of how well your model generalizes.

**Important note:** Sometimes, you might not have a separate test set and only use a dev set, especially in early stages or smaller projects. However, having a test set is best practice to avoid overfitting your model selection to the dev set.

#### Mismatched Train/Test Distribution

A common challenge is when the training data distribution differs from the dev/test data distribution. For example, training on cat pictures from the web but testing on cat pictures taken by users of your app. This mismatch can cause your model to perform poorly in real-world scenarios, so it’s important to be aware of and try to minimize this gap.


### 2. ⚖️ Understanding Bias and Variance

Bias and variance are fundamental concepts that help diagnose model performance issues.

- **Bias:** Refers to errors due to overly simplistic assumptions in the model. High bias means the model is too simple to capture the underlying patterns (underfitting).
- **Variance:** Refers to errors due to the model being too sensitive to small fluctuations in the training data. High variance means the model fits the training data too closely but fails to generalize (overfitting).

#### Diagnosing Bias and Variance

- If both training and dev errors are high, the model has **high bias**.
- If training error is low but dev error is high, the model has **high variance**.
- The goal is to find a balance where both errors are low, indicating the model is “just right.”

Example: In cat classification, if your model has high bias, it might misclassify many cats even on the training set. If it has high variance, it performs well on training images but poorly on new cat images.


### 3. 🛠️ Basic Recipe for Machine Learning

Improving a deep neural network is an iterative process involving:

1. **Idea:** Formulate a hypothesis or improvement.
2. **Experiment:** Implement changes such as adjusting the number of layers, hidden units, learning rates, or activation functions.
3. **Code:** Train the model and evaluate on dev/test sets.
4. **Repeat:** Based on results, refine your approach.

This cycle is repeated many times to gradually improve model performance.


### 4. 🛡️ Regularizing Your Neural Network

Regularization techniques help prevent overfitting by constraining the model’s complexity or adding noise during training.

#### Why Regularization Works

Regularization reduces overfitting by discouraging the model from fitting noise in the training data. It helps the model generalize better to unseen data.

- **High bias:** Regularization might worsen performance by making the model too simple.
- **Just right:** Regularization helps maintain good generalization.
- **High variance:** Regularization reduces overfitting and improves dev/test performance.

#### Common Regularization Methods

- **L2 Regularization (Weight Decay):** Adds a penalty proportional to the square of the weights to the loss function, encouraging smaller weights.
- **Dropout Regularization:** Randomly “drops out” (sets to zero) a fraction of neurons during training, forcing the network to not rely on any single feature and spread out the learned weights.
- **Data Augmentation:** Artificially increases the size of the training set by applying transformations (e.g., rotations, flips) to existing data.
- **Early Stopping:** Stops training when the dev set error starts to increase, preventing overfitting.

#### Dropout Details

- **Implementation:** During training, each neuron is kept with probability *p* and dropped with probability *1-p*. At test time, all neurons are used but their outputs are scaled by *p* (inverted dropout).
- **Why it works:** By randomly dropping neurons, the network cannot rely on any single feature, which encourages redundancy and robustness in learned features.


### 5. ⚙️ Setting Up the Optimization Problem

Training a neural network involves minimizing a cost function by adjusting weights using optimization algorithms.

#### Normalizing Inputs

- **Why normalize?** Features with different scales can cause slow or unstable training.
- **Unnormalized inputs:** Features vary widely in scale, causing gradients to be uneven.
- **Normalized inputs:** Features are scaled to have zero mean and unit variance, which helps gradients flow better and speeds up convergence.

#### Vanishing and Exploding Gradients

- **Vanishing gradients:** Gradients become very small during backpropagation, causing slow learning or no learning in early layers.
- **Exploding gradients:** Gradients become very large, causing unstable updates.
- These problems are common in deep networks and require careful initialization, normalization, or specialized architectures to mitigate.


### 6. ✅ Checking Your Derivative Computation (Gradient Checking)

Gradient checking is a debugging technique to verify that your backpropagation implementation is correct.

- **How it works:** Numerically approximate gradients by slightly perturbing parameters and comparing the numerical gradient to the one computed by backpropagation.
- **Implementation notes:**
  - Only use gradient checking for debugging, not during training.
  - It doesn’t work with dropout enabled.
  - Run gradient checks at initialization and possibly after some training.
  - If the check fails, inspect components of your code to find bugs.


### 7. 🚀 Optimization Algorithms

Optimization algorithms update model parameters to minimize the cost function efficiently.

#### Gradient Descent Variants

- **Batch Gradient Descent:** Uses the entire training set to compute gradients. Accurate but slow for large datasets.
- **Mini-batch Gradient Descent:** Uses small batches (e.g., 64 or 128 examples) to compute gradients. Balances speed and accuracy.
- **Stochastic Gradient Descent (SGD):** Uses one example at a time. Fast but noisy.

#### Vectorization

Vectorization allows efficient computation of gradients and updates over mini-batches using matrix operations, speeding up training.

#### Advanced Optimization Algorithms

- **Exponentially Weighted Averages:** Smooth out noisy gradient estimates by averaging gradients over time with decay.
- **Gradient Descent with Momentum:** Accelerates gradient descent by adding a fraction of the previous update to the current update, helping escape shallow minima and speed convergence.
- **RMSprop:** Adapts learning rates for each parameter by dividing the gradient by a running average of recent magnitudes.
- **Adam:** Combines momentum and RMSprop, maintaining running averages of both gradients and squared gradients, often leading to faster convergence.

#### Learning Rate Decay

Gradually reducing the learning rate during training helps the model converge more smoothly and avoid overshooting minima.


### 8. 🔧 Hyperparameter Tuning

Hyperparameters are settings like learning rate, batch size, number of layers, and regularization strength that control the training process.

#### Tuning Process

- Try random values rather than grid search to explore hyperparameter space more efficiently.
- Start with coarse ranges and then fine-tune around promising values.
- Use appropriate scales (e.g., logarithmic scale for learning rates).
- Re-test hyperparameters occasionally as model or data changes.


### 9. 🧮 Batch Normalization

Batch normalization is a technique to normalize the activations of each layer during training.

#### Why Batch Norm Works

- It reduces the problem of **internal covariate shift**, where the distribution of inputs to each layer changes during training.
- Normalizing activations stabilizes and speeds up training.
- Each mini-batch is normalized by its mean and variance, adding some noise similar to dropout, which also has a slight regularization effect.

#### Implementing Batch Norm

- Insert batch norm layers after linear transformations and before activation functions.
- During training, compute mean and variance per mini-batch.
- During testing, use running averages of mean and variance.


### 10. 🖥️ Programming Frameworks for Deep Learning

There are many frameworks to help build and train deep neural networks efficiently:

- **Popular frameworks:** TensorFlow, PyTorch, Keras, Caffe, MXNet, PaddlePaddle, Theano, Torch, CNTK.
- **Choosing a framework:** Consider ease of programming, speed, community support, and open-source governance.
- **TensorFlow:** Widely used, supports deployment on various platforms, and has a large ecosystem.


### Summary

Improving deep neural networks is a complex but manageable process involving careful data setup, understanding bias and variance, applying regularization, optimizing training with advanced algorithms, tuning hyperparameters, and using normalization techniques like batch norm. Debugging tools like gradient checking and efficient programming frameworks help streamline development. Iteration and experimentation are key to success.



<br>

## Questions



#### 1. What is the primary purpose of having separate training, development (dev), and test sets in a machine learning project?  
A) To increase the size of the dataset  
B) To evaluate model performance on unseen data and tune hyperparameters without bias  
C) To reduce the training time of the model  
D) To ensure the model memorizes the training data perfectly  


#### 2. Which of the following scenarios best illustrates a mismatch between training and test data distributions?  
A) Training on cat images from the web, testing on cat images taken by app users  
B) Training and testing on the same dataset  
C) Training on images of dogs, testing on images of cats  
D) Training on grayscale images, testing on grayscale images  


#### 3. If a model has low training error but high dev error, what is the most likely problem?  
A) High bias  
B) High variance  
C) Data mismatch  
D) Insufficient training data  


#### 4. Which of the following changes would most likely reduce high variance in a neural network?  
A) Increasing the number of hidden units  
B) Adding dropout regularization  
C) Decreasing the size of the training set  
D) Increasing the learning rate  


#### 5. How does dropout regularization help prevent overfitting?  
A) By permanently removing neurons from the network  
B) By forcing the network to rely on multiple features rather than any single one  
C) By increasing the model’s capacity  
D) By adding noise to the input data  


#### 6. Why is normalizing input features important before training a neural network?  
A) It ensures all features have the same units  
B) It speeds up convergence by preventing uneven gradient scales  
C) It prevents vanishing gradients entirely  
D) It guarantees the model will not overfit  


#### 7. Which of the following statements about vanishing and exploding gradients is true?  
A) Vanishing gradients cause very large weight updates  
B) Exploding gradients cause very small weight updates  
C) Both problems can slow down or destabilize training in deep networks  
D) They only occur in networks with ReLU activations  


#### 8. What is the main purpose of gradient checking in neural network training?  
A) To speed up training by approximating gradients  
B) To verify the correctness of backpropagation implementation  
C) To replace dropout during training  
D) To optimize hyperparameters automatically  


#### 9. Which of the following is NOT a recommended practice when performing gradient checking?  
A) Running gradient checks only during debugging  
B) Using gradient checking with dropout enabled  
C) Comparing numerical gradients with backpropagation gradients  
D) Running gradient checks at random initialization  


#### 10. Mini-batch gradient descent is preferred over batch gradient descent because:  
A) It always converges to the global minimum  
B) It balances computational efficiency and gradient accuracy  
C) It uses the entire dataset for every update  
D) It eliminates the need for vectorization  


#### 11. Exponentially weighted averages are used in optimization algorithms primarily to:  
A) Smooth out noisy gradient estimates over time  
B) Increase the learning rate dynamically  
C) Store all past gradients for exact averaging  
D) Replace the need for momentum  


#### 12. Which of the following best describes the Adam optimization algorithm?  
A) It uses only momentum to accelerate gradient descent  
B) It combines momentum and RMSprop to adapt learning rates and accelerate convergence  
C) It requires manual learning rate decay to work properly  
D) It is slower than vanilla gradient descent in most cases  


#### 13. Learning rate decay is useful because:  
A) It increases the learning rate over time to speed up training  
B) It helps the model converge smoothly by reducing step sizes as training progresses  
C) It prevents the model from ever reaching a minimum  
D) It is only useful when training with batch gradient descent  


#### 14. When tuning hyperparameters, why is random search often preferred over grid search?  
A) Random search explores the hyperparameter space more efficiently  
B) Grid search always finds the global optimum  
C) Random search requires fewer experiments to find good values  
D) Grid search is only useful for categorical hyperparameters  


#### 15. Batch normalization helps training by:  
A) Normalizing inputs to each layer to reduce internal covariate shift  
B) Adding noise to activations similar to dropout, providing regularization  
C) Eliminating the need for activation functions  
D) Making the network invariant to input scale changes  


#### 16. Which of the following is a potential downside of batch normalization?  
A) It requires computing mean and variance per mini-batch, adding computational overhead  
B) It always increases training time significantly  
C) It cannot be used with mini-batch gradient descent  
D) It removes the need for any other regularization method  


#### 17. Which of the following statements about bias and variance is FALSE?  
A) High bias models underfit the training data  
B) High variance models perform well on training data but poorly on dev data  
C) Increasing model complexity always reduces bias and variance simultaneously  
D) Regularization can help reduce variance but may increase bias  


#### 18. Why might you re-test hyperparameters occasionally during model development?  
A) Because intuitions and data distributions can change over time  
B) Because hyperparameters degrade during training  
C) Because once set, hyperparameters cannot be changed  
D) Because re-testing hyperparameters always improves training speed  


#### 19. Which of the following is NOT a typical characteristic of a well-regularized neural network?  
A) Low training error and low dev error  
B) High training error and high dev error  
C) Slightly higher training error than an overfitted model  
D) Better generalization to unseen data  


#### 20. In the context of deep learning frameworks, which factors are most important when choosing one?  
A) Ease of programming and deployment  
B) Running speed and scalability  
C) Open-source governance and community support  
D) The number of pre-trained models included by default  



<br>

## Answers



#### 1. What is the primary purpose of having separate training, development (dev), and test sets in a machine learning project?  
A) ✗ Increasing dataset size is not the main purpose.  
B) ✓ To evaluate model performance on unseen data and tune hyperparameters without bias.  
C) ✗ It does not directly reduce training time.  
D) ✗ The goal is generalization, not memorization.  

**Correct:** B


#### 2. Which of the following scenarios best illustrates a mismatch between training and test data distributions?  
A) ✓ Training on web cat images, testing on user cat images shows distribution mismatch.  
B) ✗ Training and testing on the same data is not a mismatch.  
C) ✗ Different classes, not distribution mismatch within the same class.  
D) ✗ Same type and format, no mismatch.  

**Correct:** A


#### 3. If a model has low training error but high dev error, what is the most likely problem?  
A) ✗ Low training error means bias is low.  
B) ✓ High variance causes overfitting to training data but poor dev performance.  
C) ✗ Data mismatch could cause issues but high variance is the primary cause here.  
D) ✗ Insufficient data usually causes high bias, not this pattern.  

**Correct:** B


#### 4. Which of the following changes would most likely reduce high variance in a neural network?  
A) ✗ Increasing hidden units usually increases variance.  
B) ✓ Dropout regularization reduces overfitting and variance.  
C) ✗ Decreasing training data size usually increases variance.  
D) ✗ Increasing learning rate can destabilize training, not reduce variance.  

**Correct:** B


#### 5. How does dropout regularization help prevent overfitting?  
A) ✗ Neurons are dropped only during training, not permanently removed.  
B) ✓ Forces network to spread weights and not rely on any single feature.  
C) ✗ Dropout reduces capacity effectively, not increases it.  
D) ✗ Dropout adds noise to activations, not input data.  

**Correct:** B


#### 6. Why is normalizing input features important before training a neural network?  
A) ✗ Normalization does not ensure same units, but scales features.  
B) ✓ Speeds up convergence by preventing uneven gradient scales.  
C) ✗ Does not completely prevent vanishing gradients.  
D) ✗ Normalization alone does not guarantee no overfitting.  

**Correct:** B


#### 7. Which of the following statements about vanishing and exploding gradients is true?  
A) ✗ Vanishing gradients cause very small updates, not large.  
B) ✗ Exploding gradients cause very large updates, not small.  
C) ✓ Both can slow or destabilize training in deep networks.  
D) ✗ They can occur with many activations, not only ReLU.  

**Correct:** C


#### 8. What is the main purpose of gradient checking in neural network training?  
A) ✗ Gradient checking is for debugging, not speeding training.  
B) ✓ Verifies correctness of backpropagation implementation.  
C) ✗ Does not replace dropout.  
D) ✗ Does not optimize hyperparameters automatically.  

**Correct:** B


#### 9. Which of the following is NOT a recommended practice when performing gradient checking?  
A) ✓ Running gradient checks only during debugging is recommended.  
B) ✗ Using gradient checking with dropout enabled is not recommended.  
C) ✓ Comparing numerical and backprop gradients is recommended.  
D) ✓ Running checks at random initialization is recommended.  

**Correct:** B


#### 10. Mini-batch gradient descent is preferred over batch gradient descent because:  
A) ✗ It does not guarantee global minimum.  
B) ✓ Balances computational efficiency and gradient accuracy.  
C) ✗ Batch gradient descent uses entire dataset, mini-batch does not.  
D) ✗ Vectorization is still needed for mini-batches.  

**Correct:** B


#### 11. Exponentially weighted averages are used in optimization algorithms primarily to:  
A) ✓ Smooth out noisy gradient estimates over time.  
B) ✗ They do not increase learning rate dynamically.  
C) ✗ They do not store all past gradients exactly.  
D) ✗ They complement momentum, not replace it.  

**Correct:** A


#### 12. Which of the following best describes the Adam optimization algorithm?  
A) ✗ Adam uses more than just momentum.  
B) ✓ Combines momentum and RMSprop for adaptive learning rates and acceleration.  
C) ✗ Adam can work without manual learning rate decay.  
D) ✗ Adam is generally faster than vanilla gradient descent.  

**Correct:** B


#### 13. Learning rate decay is useful because:  
A) ✗ It reduces, not increases, learning rate over time.  
B) ✓ Helps model converge smoothly by reducing step sizes during training.  
C) ✗ It does not prevent reaching a minimum, it helps find it better.  
D) ✗ Useful with all gradient descent variants, not only batch.  

**Correct:** B


#### 14. When tuning hyperparameters, why is random search often preferred over grid search?  
A) ✓ Explores hyperparameter space more efficiently.  
B) ✗ Grid search does not guarantee global optimum.  
C) ✓ Requires fewer experiments to find good values.  
D) ✗ Grid search works for continuous and categorical parameters.  

**Correct:** A,C


#### 15. Batch normalization helps training by:  
A) ✓ Normalizing inputs to each layer to reduce internal covariate shift.  
B) ✓ Adds noise to activations similar to dropout, providing regularization.  
C) ✗ Does not eliminate need for activation functions.  
D) ✗ Does not make network invariant to input scale changes.  

**Correct:** A,B


#### 16. Which of the following is a potential downside of batch normalization?  
A) ✓ Computing mean and variance per mini-batch adds overhead.  
B) ✗ It usually speeds up training, not increases time significantly.  
C) ✗ It is designed to work with mini-batch gradient descent.  
D) ✗ It does not remove need for other regularization methods.  

**Correct:** A


#### 17. Which of the following statements about bias and variance is FALSE?  
A) ✗ High bias models underfit training data (true).  
B) ✗ High variance models perform well on training but poorly on dev (true).  
C) ✓ Increasing model complexity does not always reduce both bias and variance simultaneously.  
D) ✗ Regularization reduces variance but may increase bias (true).  

**Correct:** C


#### 18. Why might you re-test hyperparameters occasionally during model development?  
A) ✓ Because intuitions and data distributions can change over time.  
B) ✗ Hyperparameters do not degrade during training.  
C) ✗ Hyperparameters can be changed anytime, not fixed.  
D) ✗ Re-testing does not always improve training speed.  

**Correct:** A


#### 19. Which of the following is NOT a typical characteristic of a well-regularized neural network?  
A) ✗ Low training and dev error is typical of well-regularized models.  
B) ✓ High training and dev error indicates underfitting, not good regularization.  
C) ✗ Slightly higher training error than overfitted model is expected.  
D) ✗ Better generalization to unseen data is expected.  

**Correct:** B


#### 20. In the context of deep learning frameworks, which factors are most important when choosing one?  
A) ✓ Ease of programming and deployment are important.  
B) ✓ Running speed and scalability matter.  
C) ✓ Open-source governance and community support are key.  
D) ✗ Number of pre-trained models is helpful but less critical than others.  

**Correct:** A,B,C

