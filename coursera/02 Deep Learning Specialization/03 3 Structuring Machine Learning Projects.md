## 3 Structuring Machine Learning Projects



### Key Points



#### 1. 🎯 Evaluation Metrics and Data Splits  
- Use a **single number evaluation metric** (e.g., accuracy, precision, recall, F1 score) to compare models.  
- Dev and test sets should reflect the **data distribution expected in the future**.  
- Dev set size must be large enough to detect differences between models.  
- Test set size must be large enough to give high confidence in overall system performance.  
- Change dev/test sets or metrics if performance on these sets does not correspond to real-world performance.

#### 2. 🤖 ML Strategy and Orthogonalization  
- ML strategy involves systematically choosing improvements like collecting more data, changing network size, or trying different optimization algorithms.  
- Orthogonalization means separating concerns: fit training set well → fit dev set well → fit test set well → perform well in real world.

#### 3. 🧠 Human-Level Performance  
- Human-level error is often used as a proxy for Bayes error (irreducible error).  
- Comparing to human-level performance helps in error analysis and understanding bias and variance.  
- Avoidable bias exists if dev error is much higher than human-level error.  
- High variance is indicated by low training error but high dev error.

#### 4. 🚀 Improving Model Performance  
- To reduce bias: train bigger models, train longer, use better optimization, add more data, tune architecture/hyperparameters.  
- To reduce variance: use regularization, collect more data, simplify the model.  
- Error analysis involves manually inspecting ~100 dev set errors to categorize and prioritize fixes.  
- Deep learning models are robust to some random label noise but mislabeled dev/test examples should be cleaned.

#### 5. 🔄 Mismatched Training and Dev/Test Data  
- Training and dev/test data can come from different distributions, causing poor generalization.  
- Address mismatch by manual error analysis, collecting more similar data, or artificial data synthesis.

#### 6. 🔗 Transfer and Multi-Task Learning  
- Transfer learning is useful when Task A has much more data than Task B, and both share the same input features.  
- Multi-task learning is useful when multiple related tasks have similar data amounts and can share lower-level features in a single model.

#### 7. 🧩 End-to-End Deep Learning  
- End-to-end learning trains a single model to map raw inputs directly to outputs without hand-designed intermediate components.  
- Pros: less manual design, lets data “speak.”  
- Cons: requires large amounts of data, may exclude useful hand-designed components.  
- Key question: Do you have enough data to learn the complex input-output mapping?



<br>

## Study Notes





### 1. 🤖 Introduction to Machine Learning Strategy

When starting a machine learning (ML) project, having a clear **strategy** is crucial. ML strategy is about making smart decisions on how to improve your model and system effectively. Without a strategy, you might waste time trying random tweaks that don’t help.

#### Why ML Strategy?

Imagine you have a model that isn’t performing well. What should you do next? There are many options:

- Collect more data or more diverse data
- Train the algorithm longer or try different optimization methods (e.g., Adam instead of plain gradient descent)
- Change the network size (bigger or smaller)
- Add regularization techniques like dropout or L2 regularization
- Experiment with different network architectures or activation functions
- Adjust the number of hidden units

The key is to **systematically** decide which idea to try next based on evidence, rather than guessing.

#### Orthogonalization

Orthogonalization means breaking down the problem into independent parts so you can tackle one issue at a time without interference. For example, you want to:

- Fit the training set well (low training error)
- Ensure the model generalizes well to the dev (validation) set
- Confirm it performs well on the test set
- Ultimately, it should work well in the real world

By separating these concerns, you can identify exactly where the problem lies.


### 2. 🎯 Setting Up Your Goal: Evaluation Metrics and Data Splits

Before building your model, you need to define **what success looks like**. This means choosing the right evaluation metric and carefully splitting your data.

#### Single Number Evaluation Metric

To compare models easily, pick a **single number** that summarizes performance. Common metrics include:

- **Accuracy:** Percentage of correct predictions
- **Precision and Recall:** Useful when classes are imbalanced (e.g., detecting cats vs. dogs)
- **F1 Score:** Harmonic mean of precision and recall, balances both
- **Error Rate:** Percentage of incorrect predictions

Choosing the right metric depends on your problem. For example, in loan approvals, false positives and false negatives have different costs, so precision and recall might be more relevant than accuracy.

#### Satisficing vs. Optimizing Metrics

Sometimes you want to **satisfice** (meet a minimum acceptable level) on one metric while **optimizing** another. For example, you might want your model to be at least 95% accurate but also minimize running time.

#### Data Splits: Training, Dev, and Test Sets

- **Training set:** Used to train the model.
- **Dev (validation) set:** Used to tune hyperparameters and make decisions.
- **Test set:** Used only once at the end to estimate real-world performance.

**Guidelines for splitting:**

- Dev and test sets should reflect the data you expect in the future.
- Dev set should be large enough to detect meaningful differences between models.
- Test set should be large enough to give high confidence in overall performance.

#### When to Change Dev/Test Sets or Metrics

If your model performs well on the dev/test sets but poorly in the real world, reconsider your metric or data splits. For example, if your dev set only contains medium-income zip codes but you want to deploy in low-income areas, your dev/test sets don’t represent your real-world data.


### 3. 🧠 Comparing to Human-Level Performance

A useful benchmark in many ML tasks is **human-level performance**. Humans are often very good at tasks like image recognition or speech understanding, so comparing your model to humans helps in several ways:

- **Getting labeled data:** Humans can label data for training.
- **Error analysis:** Understanding why humans get something right or wrong can guide improvements.
- **Bias/variance analysis:** Helps identify if your model is underfitting or overfitting.

#### Avoidable Bias and Variance

- **Bias:** Error due to wrong assumptions or underfitting (model too simple).
- **Variance:** Error due to overfitting (model too complex, fits noise).

If your model’s error is much higher than human-level error, you likely have **avoidable bias**. If your training error is low but dev error is high, you have **high variance**.

#### Human-Level Error as a Proxy for Bayes Error

Bayes error is the irreducible error — the best possible error any model can achieve. Human-level error often approximates this because humans are very good at many tasks.

For example, in medical image classification:

- Typical human error might be 3%
- Experienced doctors might have 0.7%
- A team of doctors might get down to 0.5%

Your model’s goal is to approach or surpass this human-level error.


### 4. 🚀 Improving Model Performance: Bias, Variance, and Error Analysis

Once you know where your model stands relative to human-level performance, you can decide how to improve it.

#### Two Fundamental Assumptions of Supervised Learning

1. You can fit the training set well (low training error).
2. Training set performance generalizes well to dev/test sets.

If these assumptions fail, you need to adjust your approach.

#### Reducing Bias and Variance

- To reduce **bias** (underfitting), try:
  - Training a bigger model
  - Training longer or using better optimization algorithms
  - Searching for better neural network architectures or hyperparameters
  - Adding more data

- To reduce **variance** (overfitting), try:
  - Regularization (dropout, L2 regularization)
  - Collecting more data
  - Simplifying the model

#### Error Analysis

Error analysis is a powerful tool where you manually inspect errors on the dev set to understand why your model is failing.

Steps:

- Collect about 100 misclassified dev examples.
- Categorize errors (e.g., dogs misclassified as cats, blurry images, mislabeled data).
- Prioritize fixes based on the most common or impactful errors.

For example, if many errors are due to dogs being labeled as cats, focus on improving dog/cat discrimination.

#### Handling Incorrect Labels

Deep learning models are somewhat robust to random label noise, but cleaning up mislabeled examples in dev/test sets is important to get accurate evaluation.


### 5. 🔄 Mismatched Training and Dev/Test Data

Sometimes, the data your model trains on differs from the data it will see in the real world (dev/test sets). This mismatch can cause poor generalization.

#### Examples of Mismatched Data

- Training on images from the web but testing on mobile app photos.
- Speech recognition trained on clean audio but tested on noisy environments.

#### Addressing Data Mismatch

- Perform manual error analysis to understand differences.
- Collect or synthesize more training data similar to dev/test sets.
- Use artificial data synthesis (e.g., adding noise, simulating environments).


### 6. 🔗 Learning from Multiple Tasks: Transfer and Multi-Task Learning

Sometimes you have related tasks that can help each other.

#### Transfer Learning

Use when:

- You have a lot of data for Task A but little for Task B.
- Both tasks share the same input features.
- Features learned from Task A can help Task B.

Example: Using a model trained on general images to help classify medical images.

#### Multi-Task Learning

Use when:

- You have multiple related tasks with similar amounts of data.
- You want to train a single model that shares lower-level features across tasks.

Example: Autonomous driving system that detects lanes, cars, and pedestrians simultaneously.


### 7. 🧩 What is End-to-End Deep Learning?

End-to-end learning means training a single model to map raw inputs directly to outputs, without breaking the problem into smaller hand-designed components.

#### Examples

- Speech recognition: raw audio → text transcription
- Face recognition: raw image → identity
- Machine translation: raw sentence in one language → sentence in another

#### Pros and Cons

**Pros:**

- The model learns directly from data, potentially discovering better features.
- Less manual feature engineering or component design.

**Cons:**

- Requires large amounts of data.
- May exclude useful domain knowledge that could be encoded in hand-designed components.

#### Key Question

Do you have enough data to learn the complex function mapping inputs to outputs? If yes, end-to-end learning can be very powerful.


### Summary

Structuring machine learning projects involves careful planning of goals, metrics, and data splits, comparing performance to human benchmarks, and systematically improving models through bias/variance analysis and error analysis. Handling data mismatches and leveraging transfer or multi-task learning can further boost performance. End-to-end deep learning offers a powerful approach when sufficient data is available.



<br>

## Questions



#### 1. What is the primary purpose of orthogonalization in structuring machine learning projects?  
A) To ensure the training error is minimized before considering dev/test errors  
B) To separate different sources of error so they can be addressed independently  
C) To combine multiple metrics into a single evaluation score  
D) To guarantee the model performs well on the real-world data  

#### 2. Which of the following are valid reasons to choose a single-number evaluation metric?  
A) Simplifies comparison between different models  
B) Captures all aspects of model performance perfectly  
C) Helps in automating model selection and tuning  
D) Avoids the need for a dev/test split  

#### 3. When might you want to change your dev/test set or evaluation metric?  
A) When the model performs well on dev/test but poorly in real-world deployment  
B) When the dev set is larger than the training set  
C) When the dev/test data distribution does not reflect future data distribution  
D) When the training error is higher than dev error  

#### 4. Why is comparing to human-level performance useful in machine learning?  
A) Humans provide a natural benchmark for achievable error rates  
B) It allows you to avoid collecting labeled data  
C) It helps in diagnosing bias and variance issues  
D) It guarantees your model will outperform humans eventually  

#### 5. Which of the following indicate high variance in a model?  
A) Low training error and high dev error  
B) High training error and high dev error  
C) Training error close to human-level error but dev error much higher  
D) Dev error lower than training error  

#### 6. What does human-level error typically approximate in machine learning tasks?  
A) The training error  
B) The Bayes error (irreducible error)  
C) The dev set error  
D) The error caused by mislabeled data  

#### 7. Which strategies can help reduce avoidable bias in a model?  
A) Training a larger neural network  
B) Adding dropout regularization  
C) Collecting more training data  
D) Training longer with better optimization algorithms  

#### 8. What is the main goal of error analysis in improving machine learning models?  
A) To identify the most common types of errors and prioritize fixes  
B) To reduce the size of the training set  
C) To find mislabeled examples in the training set only  
D) To evaluate the model’s performance on the test set  

#### 9. Which of the following are true about mislabeled data in training and dev/test sets?  
A) Deep learning models are robust to some random label noise in training data  
B) Mislabeled examples in dev/test sets can distort evaluation metrics  
C) Cleaning mislabeled examples in dev/test sets is unnecessary if training data is clean  
D) Label noise always causes overfitting  

#### 10. What is a key guideline when building your first machine learning system?  
A) Spend most time designing the perfect architecture before training  
B) Build a simple initial system quickly and iterate based on error analysis  
C) Avoid using a dev set until the final model is ready  
D) Focus only on reducing training error initially  

#### 11. Which of the following scenarios illustrate mismatched training and dev/test data distributions?  
A) Training on web images but testing on mobile app photos  
B) Training and testing on the same dataset split randomly  
C) Training on clean speech audio but testing on noisy café recordings  
D) Training on labeled data but testing on unlabeled data  

#### 12. How can you address mismatched data distributions between training and dev/test sets?  
A) Collect more training data similar to dev/test sets  
B) Use artificial data synthesis to simulate dev/test conditions  
C) Ignore the mismatch and rely on regularization  
D) Perform manual error analysis to understand differences  

#### 13. When does transfer learning make the most sense?  
A) When Task A has much more data than Task B  
B) When Task A and Task B have different input feature spaces  
C) When low-level features learned from Task A can help Task B  
D) When the tasks are completely unrelated  

#### 14. Multi-task learning is most effective when:  
A) The tasks share similar input features and have roughly equal amounts of data  
B) One task has significantly more data than the others  
C) You want to train separate models for each task independently  
D) You can train a large enough neural network to handle all tasks simultaneously  

#### 15. What is a defining characteristic of end-to-end deep learning?  
A) The model learns to map raw inputs directly to outputs without intermediate hand-designed components  
B) It requires minimal data to train effectively  
C) It always outperforms models with hand-engineered features  
D) It may exclude useful domain knowledge encoded in traditional pipelines  

#### 16. Which of the following are potential disadvantages of end-to-end learning?  
A) Requires large amounts of labeled data  
B) Can be less interpretable than modular systems  
C) Always requires manual feature engineering  
D) May not leverage useful hand-designed components  

#### 17. Why is it important to set your dev set size large enough?  
A) To detect meaningful differences between models or algorithms you try  
B) To reduce training time  
C) To ensure the dev set error is always lower than training error  
D) To guarantee the test set is unnecessary  

#### 18. What does satisficing a metric mean in the context of ML project goals?  
A) Optimizing a metric to its absolute best value  
B) Meeting a minimum acceptable threshold on one metric while optimizing another  
C) Ignoring metrics that are hard to improve  
D) Using multiple metrics without prioritizing any  

#### 19. In the context of bias and variance, what does it mean if your training error is close to human-level error but your dev error is much higher?  
A) The model has high bias  
B) The model has high variance  
C) The model is underfitting  
D) The model is overfitting  

#### 20. Which of the following are true about the chain of assumptions in supervised learning?  
A) You can fit the training set well on the cost function  
B) Training set performance generalizes well to dev/test sets  
C) Dev/test set performance guarantees real-world performance  
D) Real-world performance is independent of training and dev/test errors  



<br>

## Answers



#### 1. What is the primary purpose of orthogonalization in structuring machine learning projects?  
A) ✓ To separate different sources of error so they can be addressed independently  
B) ✓ To ensure the training error is minimized before considering dev/test errors (part of the process)  
C) ✗ To combine multiple metrics into a single evaluation score (orthogonalization separates concerns, not combines)  
D) ✗ To guarantee the model performs well on the real-world data (orthogonalization helps analysis but doesn’t guarantee)  

**Correct:** A, B


#### 2. Which of the following are valid reasons to choose a single-number evaluation metric?  
A) ✓ Simplifies comparison between different models  
B) ✗ Captures all aspects of model performance perfectly (no single metric is perfect)  
C) ✓ Helps in automating model selection and tuning  
D) ✗ Avoids the need for a dev/test split (splits are still necessary)  

**Correct:** A, C


#### 3. When might you want to change your dev/test set or evaluation metric?  
A) ✓ When the model performs well on dev/test but poorly in real-world deployment  
B) ✗ When the dev set is larger than the training set (size alone is not a reason)  
C) ✓ When the dev/test data distribution does not reflect future data distribution  
D) ✗ When the training error is higher than dev error (usually indicates other issues, not metric change)  

**Correct:** A, C


#### 4. Why is comparing to human-level performance useful in machine learning?  
A) ✓ Humans provide a natural benchmark for achievable error rates  
B) ✗ It allows you to avoid collecting labeled data (humans are needed for labeling)  
C) ✓ It helps in diagnosing bias and variance issues  
D) ✗ It guarantees your model will outperform humans eventually (no guarantee)  

**Correct:** A, C


#### 5. Which of the following indicate high variance in a model?  
A) ✓ Low training error and high dev error  
B) ✗ High training error and high dev error (indicates high bias)  
C) ✓ Training error close to human-level error but dev error much higher  
D) ✗ Dev error lower than training error (usually not possible, indicates data issues)  

**Correct:** A, C


#### 6. What does human-level error typically approximate in machine learning tasks?  
A) ✗ The training error (training error is usually lower)  
B) ✓ The Bayes error (irreducible error)  
C) ✗ The dev set error (dev error includes avoidable errors)  
D) ✗ The error caused by mislabeled data (label noise is separate)  

**Correct:** B


#### 7. Which strategies can help reduce avoidable bias in a model?  
A) ✓ Training a larger neural network  
B) ✗ Adding dropout regularization (usually reduces variance, not bias)  
C) ✓ Collecting more training data  
D) ✓ Training longer with better optimization algorithms  

**Correct:** A, C, D


#### 8. What is the main goal of error analysis in improving machine learning models?  
A) ✓ To identify the most common types of errors and prioritize fixes  
B) ✗ To reduce the size of the training set (not a goal)  
C) ✗ To find mislabeled examples in the training set only (also dev/test sets matter)  
D) ✗ To evaluate the model’s performance on the test set (error analysis is usually on dev set)  

**Correct:** A


#### 9. Which of the following are true about mislabeled data in training and dev/test sets?  
A) ✓ Deep learning models are robust to some random label noise in training data  
B) ✓ Mislabeled examples in dev/test sets can distort evaluation metrics  
C) ✗ Cleaning mislabeled examples in dev/test sets is unnecessary if training data is clean (dev/test must be clean for reliable evaluation)  
D) ✗ Label noise always causes overfitting (not always; depends on noise type)  

**Correct:** A, B


#### 10. What is a key guideline when building your first machine learning system?  
A) ✗ Spend most time designing the perfect architecture before training (wastes time)  
B) ✓ Build a simple initial system quickly and iterate based on error analysis  
C) ✗ Avoid using a dev set until the final model is ready (dev set is needed early)  
D) ✗ Focus only on reducing training error initially (need to consider generalization)  

**Correct:** B


#### 11. Which of the following scenarios illustrate mismatched training and dev/test data distributions?  
A) ✓ Training on web images but testing on mobile app photos  
B) ✗ Training and testing on the same dataset split randomly (no mismatch)  
C) ✓ Training on clean speech audio but testing on noisy café recordings  
D) ✗ Training on labeled data but testing on unlabeled data (testing requires labels for evaluation)  

**Correct:** A, C


#### 12. How can you address mismatched data distributions between training and dev/test sets?  
A) ✓ Collect more training data similar to dev/test sets  
B) ✓ Use artificial data synthesis to simulate dev/test conditions  
C) ✗ Ignore the mismatch and rely on regularization (won’t fix distribution mismatch)  
D) ✓ Perform manual error analysis to understand differences  

**Correct:** A, B, D


#### 13. When does transfer learning make the most sense?  
A) ✓ When Task A has much more data than Task B  
B) ✗ When Task A and Task B have different input feature spaces (transfer learning assumes similar inputs)  
C) ✓ When low-level features learned from Task A can help Task B  
D) ✗ When the tasks are completely unrelated (transfer unlikely to help)  

**Correct:** A, C


#### 14. Multi-task learning is most effective when:  
A) ✓ The tasks share similar input features and have roughly equal amounts of data  
B) ✗ One task has significantly more data than the others (imbalanced data reduces effectiveness)  
C) ✗ You want to train separate models for each task independently (contradicts multi-task learning)  
D) ✓ You can train a large enough neural network to handle all tasks simultaneously  

**Correct:** A, D


#### 15. What is a defining characteristic of end-to-end deep learning?  
A) ✓ The model learns to map raw inputs directly to outputs without intermediate hand-designed components  
B) ✗ It requires minimal data to train effectively (usually needs large data)  
C) ✗ It always outperforms models with hand-engineered features (not guaranteed)  
D) ✓ It may exclude useful domain knowledge encoded in traditional pipelines  

**Correct:** A, D


#### 16. Which of the following are potential disadvantages of end-to-end learning?  
A) ✓ Requires large amounts of labeled data  
B) ✓ Can be less interpretable than modular systems  
C) ✗ Always requires manual feature engineering (opposite is true)  
D) ✓ May not leverage useful hand-designed components  

**Correct:** A, B, D


#### 17. Why is it important to set your dev set size large enough?  
A) ✓ To detect meaningful differences between models or algorithms you try  
B) ✗ To reduce training time (dev set size does not affect training time)  
C) ✗ To ensure the dev set error is always lower than training error (usually dev error is higher)  
D) ✗ To guarantee the test set is unnecessary (test set is still needed)  

**Correct:** A


#### 18. What does satisficing a metric mean in the context of ML project goals?  
A) ✗ Optimizing a metric to its absolute best value (that’s optimizing)  
B) ✓ Meeting a minimum acceptable threshold on one metric while optimizing another  
C) ✗ Ignoring metrics that are hard to improve (not recommended)  
D) ✗ Using multiple metrics without prioritizing any (satisficing involves prioritization)  

**Correct:** B


#### 19. In the context of bias and variance, what does it mean if your training error is close to human-level error but your dev error is much higher?  
A) ✗ The model has high bias (high bias means high training error)  
B) ✓ The model has high variance (overfitting to training data)  
C) ✗ The model is underfitting (underfitting shows high training error)  
D) ✗ The model is overfitting (overfitting is a form of high variance, but B is more precise)  

**Correct:** B


#### 20. Which of the following are true about the chain of assumptions in supervised learning?  
A) ✓ You can fit the training set well on the cost function  
B) ✓ Training set performance generalizes well to dev/test sets  
C) ✗ Dev/test set performance guarantees real-world performance (only an assumption, not guaranteed)  
D) ✗ Real-world performance is independent of training and dev/test errors (it depends on them)  

**Correct:** A, B

