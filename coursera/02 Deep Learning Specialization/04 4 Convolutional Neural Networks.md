## 4 Convolutional Neural Networks



### Key Points



#### 1. 🧠 Convolutional Neural Networks (CNNs) Basics
- CNNs are designed to process grid-like data such as images.
- CNNs use convolutional layers, pooling layers, and fully connected layers.
- CNNs automatically learn spatial hierarchies of features through filters.

#### 2. 🖼️ Convolution Operation
- Convolution involves sliding a filter over an image and computing dot products.
- Filters detect features like edges (vertical, horizontal).
- Padding can be "valid" (no padding) or "same" (padding to keep output size equal to input).
- Stride controls how many pixels the filter moves each step.
- Convolutions extend to volumes (e.g., RGB images) by having filters with depth equal to input channels.
- Multiple filters produce multiple feature maps stacked as output.

#### 3. 🧩 CNN Layers and Parameters
- Number of parameters in a convolutional layer = (#filters) × (filter height × filter width × filter depth + 1 bias).
- Pooling layers reduce spatial dimensions using max pooling or average pooling.
- Pooling hyperparameters: filter size (f) and stride (s).
- Fully connected layers come after convolution and pooling layers for classification.

#### 4. ⚙️ Advantages of Convolutions
- Parameter sharing: same filter used across the image reduces parameters.
- Sparsity of connections: each output depends on a small local region of input.

#### 5. 🏛️ Classic CNN Architectures
- LeNet-5: early CNN for digit recognition using average pooling.
- AlexNet: introduced ReLU, max pooling, won ImageNet 2012.
- VGG-16: uses 3×3 filters, stride 1, same padding, max pooling 2×2 stride 2.
- ResNet: uses residual blocks with skip connections to train very deep networks.
- Inception: uses parallel convolutions of different sizes and pooling concatenated.
- MobileNet: uses depthwise separable convolutions to reduce computation.

#### 6. 📉 Pooling and Strided Convolutions
- Pooling reduces spatial size and computation.
- Strided convolution reduces output size by moving filter more than one pixel at a time.

#### 7. 🚀 Advanced CNN Techniques
- Residual blocks add input to output to ease training of deep networks.
- 1×1 convolutions reduce channel dimensions and add non-linearity.
- Inception modules combine multiple filter sizes and pooling in parallel.
- Depthwise separable convolutions split normal convolution into depthwise and pointwise convolutions, reducing cost.

#### 8. 🎯 Object Detection Concepts
- Object detection predicts class and bounding box coordinates.
- Sliding window detection can be implemented as convolutional layers.
- YOLO divides image into grid cells, predicts bounding boxes and class probabilities per cell.
- Non-max suppression removes overlapping bounding boxes based on IoU threshold.
- Intersection over Union (IoU) ≥ 0.5 is considered a correct detection.
- Anchor boxes allow multiple bounding boxes per grid cell.

#### 9. 🧩 Semantic Segmentation and U-Net
- Semantic segmentation assigns class labels to each pixel.
- U-Net uses encoder-decoder architecture with skip connections.
- Transpose convolutions are used for upsampling in segmentation.

#### 10. 🧑‍🤝‍🧑 Face Recognition
- Face verification: confirm if input image matches claimed identity.
- Face recognition: identify person from a database.
- One-shot learning uses similarity functions to compare images.
- Siamese networks learn embeddings where same-person images are close.
- Triplet loss trains on anchor, positive, negative images to separate classes.

#### 11. 🎨 Neural Style Transfer
- Combines content of one image with style of another.
- Content cost measures similarity of high-level features between generated and content images.
- Style cost measures correlations (Gram matrices) between feature maps to capture style.
- Gradient descent optimizes generated image to minimize combined content and style cost.

#### 12. 🔍 Visualizing CNN Layers
- Early layers detect edges and colors.
- Middle layers detect textures and parts.
- Deep layers detect complex objects and semantic features.

#### 13. 📊 Practical CNN Tips
- Transfer learning: fine-tune pre-trained networks.
- Data augmentation: mirroring, cropping, rotation, color shifts.
- Ensembling: average outputs of multiple models.
- Multi-crop testing: average predictions over multiple image crops.



<br>

## Study Notes



### Study Notes: Convolutional Neural Networks and Applications


### 1. 🧠 Introduction to Convolutional Neural Networks (CNNs)

Convolutional Neural Networks (CNNs) are a specialized type of deep learning model designed primarily for processing data that has a grid-like topology, such as images. Unlike traditional neural networks, CNNs are particularly effective for computer vision tasks because they can automatically and adaptively learn spatial hierarchies of features through backpropagation by using multiple building blocks such as convolution layers, pooling layers, and fully connected layers.

#### Why CNNs for Computer Vision?

Computer vision problems involve understanding and interpreting images or videos. Common tasks include:

- **Image classification:** Determining what object is in an image (e.g., cat or no cat).
- **Object detection:** Identifying and locating objects within an image.
- **Neural style transfer:** Combining the content of one image with the style of another.
- **Semantic segmentation:** Assigning a class label to every pixel in an image.

CNNs excel at these tasks because they can detect edges, shapes, textures, and complex patterns in images by learning filters that respond to specific visual features.


### 2. 🖼️ Understanding Convolutions in CNNs

At the heart of CNNs is the **convolution operation**, which is a mathematical way to extract features from images.

#### What is a Convolution?

A convolution involves sliding a small matrix called a **filter** or **kernel** over the input image and computing dot products between the filter and the overlapping regions of the image. This operation produces a **feature map** that highlights certain features like edges or textures.

- **Edge detection example:** Filters can be designed to detect vertical or horizontal edges by emphasizing changes in pixel intensity in those directions.
- **Learning edges:** CNNs learn these filters automatically during training, starting from simple edge detectors in early layers to more complex patterns in deeper layers.

#### Padding and Stride

- **Padding:** Sometimes, to preserve the spatial size of the input after convolution, we add extra pixels (usually zeros) around the border of the image.  
  - **Valid padding:** No padding; output size shrinks.  
  - **Same padding:** Padding added so that output size equals input size.

- **Stride:** The number of pixels the filter moves each time it slides over the image. A stride of 1 means the filter moves one pixel at a time; larger strides reduce the output size.

#### Convolutions on Volumes

Images often have multiple channels (e.g., RGB images have 3 channels). Convolutions extend to these volumes by having filters with the same depth as the input volume. Multiple filters produce multiple feature maps, which are stacked to form the output volume.


### 3. 🧩 Layers in a Convolutional Neural Network

A CNN is composed of several types of layers, each serving a specific purpose:

#### Convolutional Layers

These layers apply multiple filters to the input to produce feature maps. Each filter detects a different feature. The number of parameters in a convolutional layer depends on the filter size, depth, and number of filters.

Example:  
If you have 10 filters of size 3×3×3 (height × width × depth), the total number of parameters is:  
10 filters × (3×3×3 weights + 1 bias) = 10 × (27 + 1) = 280 parameters.

#### Pooling Layers

Pooling layers reduce the spatial size of the feature maps, which helps reduce computation and controls overfitting.

- **Max pooling:** Takes the maximum value in each region.
- **Average pooling:** Takes the average value in each region.

Pooling layers have hyperparameters like filter size (f) and stride (s).

#### Fully Connected Layers

At the end of the network, fully connected layers take the flattened feature maps and output class scores or other predictions.


### 4. ⚙️ Why Use Convolutions? Key Advantages

CNNs are powerful because of two main properties:

- **Parameter sharing:** The same filter (feature detector) is used across the entire image, which reduces the number of parameters and allows the network to detect the same feature anywhere in the image.
- **Sparsity of connections:** Each output value depends only on a small region of the input (the receptive field), making computations efficient and focusing on local features.


### 5. 🏛️ Classic CNN Architectures: Case Studies

Several landmark CNN architectures have shaped the field of computer vision:

- **LeNet-5 (1998):** One of the first CNNs, designed for digit recognition. It used convolutional and average pooling layers.
- **AlexNet (2012):** Popularized deep CNNs with ReLU activations and max pooling, winning the ImageNet challenge.
- **VGG-16 (2015):** Used very small 3×3 filters stacked deeply, with max pooling layers, showing that depth improves performance.
- **ResNet (2015):** Introduced residual connections (skip connections) to allow very deep networks to train effectively by mitigating the vanishing gradient problem.
- **Inception Network:** Uses multiple filter sizes in parallel within the same layer to capture different types of features efficiently.
- **MobileNet:** Designed for mobile and embedded devices, uses depthwise separable convolutions to reduce computational cost drastically.


### 6. 📉 Pooling and Strided Convolutions: Reducing Dimensions

Pooling layers and strided convolutions are techniques to reduce the spatial dimensions of feature maps:

- **Pooling:** Downsamples the input by summarizing regions (max or average).
- **Strided convolution:** Instead of moving the filter one pixel at a time, it moves by multiple pixels (stride > 1), reducing output size.

These operations help reduce computational load and increase the receptive field of neurons in deeper layers.


### 7. 🚀 Advanced CNN Techniques and Architectures

#### Residual Networks (ResNets)

ResNets introduced **residual blocks** where the input to a layer is added directly to its output. This "skip connection" helps gradients flow through very deep networks, allowing training of hundreds of layers without degradation.

#### 1×1 Convolutions and Network in Network

1×1 convolutions act like feature-wise linear combinations, allowing the network to:

- Reduce the number of channels (dimensionality reduction).
- Add non-linearity between layers.
- Increase the depth of the network without increasing spatial dimensions.

#### Inception Modules

Inception modules combine multiple convolutions of different sizes (1×1, 3×3, 5×5) and pooling in parallel, concatenating their outputs. This captures features at multiple scales efficiently.

#### MobileNets and Depthwise Separable Convolutions

MobileNets use **depthwise separable convolutions**, which split a normal convolution into:

- **Depthwise convolution:** Applies a single filter per input channel.
- **Pointwise convolution:** Uses 1×1 convolutions to combine the outputs.

This reduces computation and parameters, making MobileNets suitable for mobile devices.


### 8. 🎯 Object Detection and Localization

Object detection involves not only classifying objects but also locating them within an image by predicting bounding boxes.

#### Key Concepts

- **Localization:** Predicting the coordinates of objects.
- **Detection:** Predicting both the class and location.

#### Sliding Window and Convolutional Implementation

Traditional sliding window methods scan the image with a fixed-size window, classifying each patch. CNNs can implement this efficiently by converting fully connected layers into convolutional layers, allowing the network to process the entire image at once.

#### YOLO (You Only Look Once)

YOLO divides the image into a grid and predicts bounding boxes and class probabilities for each grid cell simultaneously. It uses:

- **Anchor boxes:** Predefined bounding boxes to handle multiple objects per grid cell.
- **Non-max suppression:** Removes overlapping boxes to keep only the best predictions.
- **Intersection over Union (IoU):** Measures overlap between predicted and ground truth boxes; predictions with IoU ≥ 0.5 are considered correct.

#### Region Proposal Networks (R-CNN Family)

- **R-CNN:** Proposes regions and classifies each separately.
- **Fast R-CNN:** Uses convolutional features for all proposals simultaneously.
- **Faster R-CNN:** Integrates region proposal into the CNN for real-time detection.


### 9. 🧩 Semantic Segmentation and U-Net

Semantic segmentation assigns a class label to every pixel, producing detailed maps of objects.

#### U-Net Architecture

U-Net is a fully convolutional network designed for biomedical image segmentation. It uses:

- **Encoder:** Downsampling path with convolutions and pooling.
- **Decoder:** Upsampling path with transpose convolutions.
- **Skip connections:** Connect encoder and decoder layers to preserve spatial information.


### 10. 🧑‍🤝‍🧑 Face Recognition and Verification

Face recognition systems identify or verify individuals from images.

- **Verification:** Given an image and claimed identity, decide if they match.
- **Recognition:** Identify the person from a database of known individuals.

#### One-shot Learning and Similarity Functions

Face recognition often uses one-shot learning, where the system learns to recognize a person from a single example by learning a similarity function that measures how alike two images are.

#### Siamese Networks

Siamese networks learn embeddings such that images of the same person are close in feature space, and images of different people are far apart.

#### Triplet Loss

Triplet loss trains the network on triplets of images: an anchor, a positive (same person), and a negative (different person). The goal is to minimize the distance between anchor and positive while maximizing the distance between anchor and negative.


### 11. 🎨 Neural Style Transfer

Neural style transfer is a technique that combines the **content** of one image with the **style** of another to create a new, artistic image.

#### How It Works

- Use a pre-trained CNN (e.g., VGG) to extract features.
- Define a **content cost function** that measures how similar the generated image is to the content image in terms of high-level features.
- Define a **style cost function** that measures how similar the textures and patterns (style) are by comparing correlations between feature maps (Gram matrices).
- Use gradient descent to iteratively update the generated image to minimize a weighted sum of content and style costs.


### 12. 🔍 Visualizing CNNs: What Are They Learning?

By visualizing activations of different layers, we can see that:

- Early layers detect simple features like edges and colors.
- Middle layers detect textures and parts of objects.
- Deeper layers detect complex objects and semantic concepts.

This helps us understand how CNNs build up hierarchical representations of images.


### 13. 📊 Practical Advice for Using CNNs

- **Transfer learning:** Use pre-trained networks and fine-tune on your dataset.
- **Data augmentation:** Improve generalization by applying transformations like mirroring, cropping, rotation, color shifts.
- **Ensembling:** Combine predictions from multiple models to improve accuracy.
- **Multi-crop testing:** Average predictions over multiple crops of the test image.


### Summary

Convolutional Neural Networks are powerful tools for image-related tasks, leveraging convolutions to detect features efficiently. Through layers of convolutions, pooling, and fully connected units, CNNs learn hierarchical representations of images. Advanced architectures like ResNet, Inception, and MobileNet improve performance and efficiency. CNNs are widely used in object detection, semantic segmentation, face recognition, and even creative tasks like neural style transfer. Understanding the building blocks and practical techniques is essential for applying CNNs effectively.



<br>

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

