## 4 Convolutional Neural Networks

## Questions



#### 1. What are the main advantages of using convolutional layers in CNNs compared to fully connected layers?  
A) Parameter sharing reduces the total number of parameters  
B) Each output depends on all input pixels, increasing model capacity  
C) Sparsity of connections focuses computation on local regions  
D) Convolutions guarantee translation invariance of the network  

#### 2. Which of the following statements about padding in convolutional layers are true?  
A) "Valid" padding means no padding is added, so output size is smaller than input  
B) "Same" padding adds zeros to keep output size equal to input size  
C) Padding always increases the number of parameters in the convolutional layer  
D) Padding affects the spatial dimensions but not the depth of the output volume  

#### 3. In a convolutional layer with 10 filters of size 3×3×3, how many parameters are there?  
A) 270  
B) 280  
C) 90  
D) 300  

#### 4. Which of the following best describe the role of 1×1 convolutions in CNN architectures?  
A) They reduce the number of channels in the input volume  
B) They increase the spatial resolution of feature maps  
C) They add non-linearity between layers without changing spatial dimensions  
D) They are used exclusively for downsampling  

#### 5. Why do residual connections in ResNets help train very deep networks?  
A) They add more parameters to increase model capacity  
B) They allow gradients to flow directly through skip connections, mitigating vanishing gradients  
C) They reduce the number of layers needed by combining multiple layers into one  
D) They enable the network to learn identity mappings easily  

#### 6. Which of the following are true about depthwise separable convolutions used in MobileNets?  
A) They split convolution into depthwise and pointwise convolutions  
B) They increase computational cost compared to normal convolutions  
C) They reduce the number of parameters and computations significantly  
D) They are only applicable to grayscale images  

#### 7. In object detection, what is the purpose of non-max suppression?  
A) To increase the number of predicted bounding boxes  
B) To remove overlapping bounding boxes with lower confidence scores  
C) To merge multiple bounding boxes into one large box  
D) To select the bounding box with the highest Intersection over Union (IoU)  

#### 8. Which of the following statements about Intersection over Union (IoU) are correct?  
A) IoU measures the overlap between predicted and ground truth bounding boxes  
B) An IoU threshold of 0.5 is commonly used to determine correct detections  
C) IoU is always between 0 and 1, where 0 means perfect overlap  
D) IoU can be greater than 1 if boxes are large  

#### 9. What is the main difference between semantic segmentation and object detection?  
A) Semantic segmentation assigns a class label to every pixel  
B) Object detection only classifies the entire image  
C) Object detection outputs bounding boxes around objects  
D) Semantic segmentation does not localize objects  

#### 10. In the context of face recognition, what is the purpose of triplet loss?  
A) To minimize the distance between embeddings of different persons  
B) To maximize the distance between embeddings of the same person  
C) To ensure the anchor-positive pair is closer than the anchor-negative pair by a margin  
D) To randomly select triplets during training for better generalization  

#### 11. Which of the following are true about the YOLO object detection algorithm?  
A) It divides the image into a grid and predicts bounding boxes for each cell  
B) It uses anchor boxes to handle multiple objects per grid cell  
C) It processes proposed regions one at a time for classification  
D) It applies non-max suppression to filter overlapping predictions  

#### 12. Regarding pooling layers, which statements are accurate?  
A) Max pooling selects the maximum value in each pooling window  
B) Average pooling is more effective than max pooling in all cases  
C) Pooling layers reduce spatial dimensions but keep the number of channels unchanged  
D) Pooling layers add trainable parameters to the network  

#### 13. Which of the following statements about convolution stride are correct?  
A) Increasing stride reduces the spatial size of the output feature map  
B) Stride controls how far the filter moves at each step during convolution  
C) Stride affects the depth of the output volume  
D) Strided convolutions can replace pooling layers for downsampling  

#### 14. What is the main motivation behind the Inception network architecture?  
A) To reduce computational cost by using multiple filter sizes in parallel  
B) To increase the depth of the network without increasing width  
C) To capture features at multiple scales simultaneously  
D) To replace convolutional layers with fully connected layers  

#### 15. Which of the following are challenges addressed by transfer learning in CNNs?  
A) Training deep networks from scratch requires large labeled datasets  
B) Transfer learning allows fine-tuning pre-trained models on smaller datasets  
C) Transfer learning eliminates the need for data augmentation  
D) Transfer learning can speed up convergence during training  

#### 16. In neural style transfer, what does the style cost function measure?  
A) The difference in pixel values between the generated and style images  
B) The correlation between activations across different channels in a layer  
C) The similarity of high-level content features between images  
D) The total variation loss to smooth the generated image  

#### 17. Which of the following statements about the use of skip connections in U-Net are true?  
A) Skip connections help preserve spatial information lost during downsampling  
B) They connect encoder layers directly to decoder layers  
C) Skip connections increase the number of parameters exponentially  
D) They improve segmentation accuracy by combining low-level and high-level features  

#### 18. When converting fully connected layers to convolutional layers for sliding window detection, which of the following are true?  
A) Fully connected layers become 1×1 convolutions  
B) This allows the network to process the entire image efficiently  
C) The spatial dimensions of the output remain fixed regardless of input size  
D) This technique is used to speed up object detection  

#### 19. Which of the following are true about parameter sharing in CNNs?  
A) The same filter is applied across different spatial locations of the input  
B) Parameter sharing increases the total number of parameters in the network  
C) It helps the network generalize better to unseen image locations  
D) Parameter sharing is only applicable to the first convolutional layer  

#### 20. Regarding data augmentation in CNN training, which statements are correct?  
A) Mirroring and random cropping help increase dataset diversity  
B) Data augmentation can reduce overfitting  
C) Color shifting is only useful for grayscale images  
D) Implementing distortions during training improves model robustness  



<br>

## Answers



#### 1. What are the main advantages of using convolutional layers in CNNs compared to fully connected layers?  
A) ✓ Parameter sharing reduces the total number of parameters  
B) ✗ Each output depends on all input pixels, increasing model capacity (Convolutions depend only on local regions, not all pixels)  
C) ✓ Sparsity of connections focuses computation on local regions  
D) ✗ Convolutions guarantee translation invariance of the network (They help but do not guarantee full invariance)

**Correct:** A, C


#### 2. Which of the following statements about padding in convolutional layers are true?  
A) ✓ "Valid" padding means no padding is added, so output size is smaller than input  
B) ✓ "Same" padding adds zeros to keep output size equal to input size  
C) ✗ Padding always increases the number of parameters in the convolutional layer (Padding affects output size, not parameters)  
D) ✓ Padding affects the spatial dimensions but not the depth of the output volume  

**Correct:** A, B, D


#### 3. In a convolutional layer with 10 filters of size 3×3×3, how many parameters are there?  
A) ✗ 270 (3×3×3=27 weights per filter, 10 filters → 270 weights, but biases also count)  
B) ✓ 280 (27 weights + 1 bias per filter × 10 filters = 280)  
C) ✗ 90 (too low, ignores depth and number of filters)  
D) ✗ 300 (overestimates by counting extra parameters)

**Correct:** B


#### 4. Which of the following best describe the role of 1×1 convolutions in CNN architectures?  
A) ✓ They reduce the number of channels in the input volume  
B) ✗ They increase the spatial resolution of feature maps (1×1 convs do not change spatial size)  
C) ✓ They add non-linearity between layers without changing spatial dimensions  
D) ✗ They are used exclusively for downsampling (They do not downsample)

**Correct:** A, C


#### 5. Why do residual connections in ResNets help train very deep networks?  
A) ✗ They add more parameters to increase model capacity (They add skip connections, not necessarily more parameters)  
B) ✓ They allow gradients to flow directly through skip connections, mitigating vanishing gradients  
C) ✗ They reduce the number of layers needed by combining multiple layers into one (They enable deeper networks)  
D) ✓ They enable the network to learn identity mappings easily  

**Correct:** B, D


#### 6. Which of the following are true about depthwise separable convolutions used in MobileNets?  
A) ✓ They split convolution into depthwise and pointwise convolutions  
B) ✗ They increase computational cost compared to normal convolutions (They reduce cost)  
C) ✓ They reduce the number of parameters and computations significantly  
D) ✗ They are only applicable to grayscale images (Applicable to any multi-channel input)

**Correct:** A, C


#### 7. In object detection, what is the purpose of non-max suppression?  
A) ✗ To increase the number of predicted bounding boxes (It reduces redundant boxes)  
B) ✓ To remove overlapping bounding boxes with lower confidence scores  
C) ✗ To merge multiple bounding boxes into one large box (It discards overlapping boxes)  
D) ✗ To select the bounding box with the highest Intersection over Union (IoU) (It selects based on confidence, not IoU)

**Correct:** B


#### 8. Which of the following statements about Intersection over Union (IoU) are correct?  
A) ✓ IoU measures the overlap between predicted and ground truth bounding boxes  
B) ✓ An IoU threshold of 0.5 is commonly used to determine correct detections  
C) ✗ IoU is always between 0 and 1, where 0 means perfect overlap (0 means no overlap, 1 means perfect overlap)  
D) ✗ IoU can be greater than 1 if boxes are large (IoU is a ratio and capped at 1)

**Correct:** A, B


#### 9. What is the main difference between semantic segmentation and object detection?  
A) ✓ Semantic segmentation assigns a class label to every pixel  
B) ✗ Object detection only classifies the entire image (It also localizes objects)  
C) ✓ Object detection outputs bounding boxes around objects  
D) ✗ Semantic segmentation does not localize objects (It localizes at pixel level)

**Correct:** A, C


#### 10. In the context of face recognition, what is the purpose of triplet loss?  
A) ✗ To minimize the distance between embeddings of different persons (It maximizes this distance)  
B) ✗ To maximize the distance between embeddings of the same person (It minimizes this distance)  
C) ✓ To ensure the anchor-positive pair is closer than the anchor-negative pair by a margin  
D) ✗ To randomly select triplets during training for better generalization (Triplets are chosen carefully, not randomly)

**Correct:** C


#### 11. Which of the following are true about the YOLO object detection algorithm?  
A) ✓ It divides the image into a grid and predicts bounding boxes for each cell  
B) ✓ It uses anchor boxes to handle multiple objects per grid cell  
C) ✗ It processes proposed regions one at a time for classification (YOLO processes the whole image at once)  
D) ✓ It applies non-max suppression to filter overlapping predictions  

**Correct:** A, B, D


#### 12. Regarding pooling layers, which statements are accurate?  
A) ✓ Max pooling selects the maximum value in each pooling window  
B) ✗ Average pooling is more effective than max pooling in all cases (Effectiveness depends on task)  
C) ✓ Pooling layers reduce spatial dimensions but keep the number of channels unchanged  
D) ✗ Pooling layers add trainable parameters to the network (Pooling has no parameters)

**Correct:** A, C


#### 13. Which of the following statements about convolution stride are correct?  
A) ✓ Increasing stride reduces the spatial size of the output feature map  
B) ✓ Stride controls how far the filter moves at each step during convolution  
C) ✗ Stride affects the depth of the output volume (Depth depends on number of filters)  
D) ✓ Strided convolutions can replace pooling layers for downsampling  

**Correct:** A, B, D


#### 14. What is the main motivation behind the Inception network architecture?  
A) ✓ To reduce computational cost by using multiple filter sizes in parallel  
B) ✗ To increase the depth of the network without increasing width (It increases width by parallel filters)  
C) ✓ To capture features at multiple scales simultaneously  
D) ✗ To replace convolutional layers with fully connected layers  

**Correct:** A, C


#### 15. Which of the following are challenges addressed by transfer learning in CNNs?  
A) ✓ Training deep networks from scratch requires large labeled datasets  
B) ✓ Transfer learning allows fine-tuning pre-trained models on smaller datasets  
C) ✗ Transfer learning eliminates the need for data augmentation (Augmentation is still useful)  
D) ✓ Transfer learning can speed up convergence during training  

**Correct:** A, B, D


#### 16. In neural style transfer, what does the style cost function measure?  
A) ✗ The difference in pixel values between the generated and style images (It compares feature correlations)  
B) ✓ The correlation between activations across different channels in a layer  
C) ✗ The similarity of high-level content features between images (That’s content cost)  
D) ✗ The total variation loss to smooth the generated image (Separate regularization term)

**Correct:** B


#### 17. Which of the following statements about the use of skip connections in U-Net are true?  
A) ✓ Skip connections help preserve spatial information lost during downsampling  
B) ✓ They connect encoder layers directly to decoder layers  
C) ✗ Skip connections increase the number of parameters exponentially (They add no parameters)  
D) ✓ They improve segmentation accuracy by combining low-level and high-level features  

**Correct:** A, B, D


#### 18. When converting fully connected layers to convolutional layers for sliding window detection, which of the following are true?  
A) ✓ Fully connected layers become 1×1 convolutions  
B) ✓ This allows the network to process the entire image efficiently  
C) ✗ The spatial dimensions of the output remain fixed regardless of input size (Output size depends on input size)  
D) ✓ This technique is used to speed up object detection  

**Correct:** A, B, D


#### 19. Which of the following are true about parameter sharing in CNNs?  
A) ✓ The same filter is applied across different spatial locations of the input  
B) ✗ Parameter sharing increases the total number of parameters in the network (It reduces parameters)  
C) ✓ It helps the network generalize better to unseen image locations  
D) ✗ Parameter sharing is only applicable to the first convolutional layer (It applies to all conv layers)

**Correct:** A, C


#### 20. Regarding data augmentation in CNN training, which statements are correct?  
A) ✓ Mirroring and random cropping help increase dataset diversity  
B) ✓ Data augmentation can reduce overfitting  
C) ✗ Color shifting is only useful for grayscale images (It’s more useful for color images)  
D) ✓ Implementing distortions during training improves model robustness  

**Correct:** A, B, D

