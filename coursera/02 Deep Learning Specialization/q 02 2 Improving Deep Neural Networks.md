## 2 Improving Deep Neural Networks

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

