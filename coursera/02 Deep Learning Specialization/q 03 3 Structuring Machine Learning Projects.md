## 3 Structuring Machine Learning Projects

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

