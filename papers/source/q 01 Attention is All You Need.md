## Attention is All You Need

## Questions

#### 1. What is the primary architectural innovation of the Transformer compared to traditional sequence models?  
A) Use of convolutional layers instead of recurrent layers  
B) Complete removal of recurrence and convolution, relying solely on attention  
C) Introduction of gated recurrent units (GRUs)  
D) Use of multi-head self-attention mechanisms  

#### 2. Which of the following are advantages of self-attention over recurrent layers?  
A) Enables parallel computation across sequence positions  
B) Requires fewer sequential operations proportional to sequence length  
C) Automatically encodes positional information without additional input  
D) Shorter maximum path length between any two positions in the sequence  

#### 3. In scaled dot-product attention, why is the dot product scaled by the square root of the key dimension?  
A) To increase the magnitude of the dot products for better gradient flow  
B) To prevent the dot products from growing too large and pushing softmax into regions with small gradients  
C) To normalize the attention weights so they sum to one  
D) To reduce computational complexity  

#### 4. Multi-head attention improves model performance primarily because:  
A) It increases the total dimensionality of the model  
B) It allows the model to attend to information from different representation subspaces simultaneously  
C) It averages attention weights to reduce noise  
D) It enables the model to capture multiple types of relationships in parallel  

#### 5. Which of the following statements about positional encoding in the Transformer are true?  
A) Positional encodings are learned parameters updated during training  
B) Sinusoidal positional encodings allow the model to generalize to longer sequences than seen during training  
C) Positional encodings are added to input embeddings to inject order information  
D) The Transformer uses recurrent positional embeddings to encode sequence order  

#### 6. How does the Transformer’s decoder prevent attending to future tokens during training?  
A) By using a masking mechanism that sets illegal attention weights to negative infinity before softmax  
B) By limiting the attention window to previous tokens only  
C) By using a separate recurrent network for the decoder  
D) By offsetting output embeddings by one position  

#### 7. Which of the following are components of each encoder layer in the Transformer?  
A) Multi-head self-attention  
B) Position-wise feed-forward network  
C) Encoder-decoder attention  
D) Residual connections and layer normalization  

#### 8. What is the role of residual connections in the Transformer architecture?  
A) To allow gradients to flow more easily through deep networks  
B) To reduce the number of parameters in the model  
C) To combine outputs of attention heads  
D) To normalize the input embeddings  

#### 9. Compared to convolutional sequence models like ConvS2S and ByteNet, self-attention layers:  
A) Have a maximum path length between positions that grows linearly with sequence length  
B) Have a constant maximum path length between any two positions  
C) Require fewer operations to relate distant positions  
D) Are less parallelizable than convolutional layers  

#### 10. Why might additive attention outperform dot-product attention without scaling for large key dimensions?  
A) Because additive attention uses a feed-forward network that better models compatibility  
B) Because dot-product attention is computationally more expensive  
C) Because dot-product attention’s unscaled dot products can become very large, causing small gradients  
D) Because additive attention normalizes the keys and queries  

#### 11. Which of the following describe the training regime used for the Transformer?  
A) Use of Adam optimizer with warm-up learning rate schedule  
B) Training on batches grouped by approximate sequence length  
C) Use of label smoothing to improve BLEU scores despite hurting perplexity  
D) Training exclusively on single GPUs for maximum efficiency  

#### 12. In the Transformer, what is the dimensionality relationship between the number of attention heads (h), the model dimension (dmodel), and the key/value dimensions (dk, dv)?  
A) \( d_k = d_v = \frac{d_{model}}{h} \)  
B) \( d_k = d_v = d_{model} \times h \)  
C) \( d_k = d_v = d_{model} \)  
D) \( d_k = d_v = \frac{h}{d_{model}} \)  

#### 13. Which of the following are true about the feed-forward networks in the Transformer layers?  
A) They are applied independently to each position in the sequence  
B) They consist of two linear transformations with a ReLU activation in between  
C) They share parameters across all layers  
D) They can be interpreted as convolutions with kernel size 1  

#### 14. How does the Transformer handle input and output token embeddings and the final softmax layer?  
A) Uses separate weight matrices for input embeddings, output embeddings, and softmax  
B) Shares the same weight matrix between input embeddings, output embeddings, and the pre-softmax linear transformation  
C) Multiplies embeddings by the square root of the model dimension before adding positional encodings  
D) Uses learned positional embeddings only for the decoder  

#### 15. What is the main reason the Transformer can be trained faster than RNN-based models?  
A) It uses fewer parameters overall  
B) It allows parallelization across all positions in the sequence during training  
C) It uses convolutional layers that are faster than recurrent layers  
D) It requires fewer training steps to converge  

#### 16. Which of the following statements about the maximum path length in different layer types is correct?  
A) Recurrent layers have a maximum path length proportional to the sequence length \(O(n)\)  
B) Self-attention layers have a maximum path length of \(O(1)\)  
C) Convolutional layers with kernel size \(k\) have maximum path length \(O(\log_k n)\) if dilated  
D) Self-attention layers have longer maximum path lengths than recurrent layers  

#### 17. In the context of the Transformer, what is the significance of masking in the decoder’s self-attention?  
A) It prevents the model from attending to padding tokens  
B) It enforces the autoregressive property by blocking attention to future tokens  
C) It improves computational efficiency by reducing the number of keys considered  
D) It is only applied during inference, not training  

#### 18. Which of the following are challenges or limitations of self-attention that the Transformer addresses?  
A) Reduced effective resolution due to averaging attention-weighted positions  
B) Difficulty in learning long-range dependencies  
C) High computational cost for very long sequences  
D) Inability to model positional information without recurrence  

#### 19. How does the Transformer generalize to tasks beyond machine translation, such as English constituency parsing?  
A) By using the same architecture with minimal task-specific tuning  
B) By adding task-specific recurrent layers to the decoder  
C) By increasing the number of attention heads significantly  
D) By training on much larger datasets only  

#### 20. Which of the following statements about the learning rate schedule used in training the Transformer are true?  
A) The learning rate increases linearly during a warm-up phase  
B) After warm-up, the learning rate decreases proportionally to the inverse square root of the step number  
C) The learning rate remains constant throughout training  
D) Warm-up steps are set to 4000 in the base model



<br>

## Answers

#### 1. What is the primary architectural innovation of the Transformer compared to traditional sequence models?  
A) ✗ The Transformer removes recurrence and convolution, not replaces them with convolution.  
B) ✓ The Transformer relies solely on attention, removing recurrence and convolution entirely.  
C) ✗ GRUs are a type of RNN, not part of the Transformer innovation.  
D) ✓ Multi-head self-attention is a key part of the Transformer’s architecture.  

**Correct:** B, D


#### 2. Which of the following are advantages of self-attention over recurrent layers?  
A) ✓ Self-attention allows parallel computation across all positions.  
B) ✓ Requires fewer sequential operations, independent of sequence length.  
C) ✗ Positional information is injected separately via positional encodings.  
D) ✓ Maximum path length between positions is constant, aiding long-range dependency learning.  

**Correct:** A, B, D


#### 3. In scaled dot-product attention, why is the dot product scaled by the square root of the key dimension?  
A) ✗ Scaling reduces magnitude, not increases it.  
B) ✓ Prevents large dot products that push softmax into regions with tiny gradients.  
C) ✗ Softmax normalization is independent of scaling factor.  
D) ✗ Scaling does not reduce computational complexity.  

**Correct:** B


#### 4. Multi-head attention improves model performance primarily because:  
A) ✗ It does not increase total model dimensionality; it splits it.  
B) ✓ Allows attending to different representation subspaces simultaneously.  
C) ✗ Averaging would reduce expressiveness; multi-head concatenates outputs.  
D) ✓ Captures multiple types of relationships in parallel.  

**Correct:** B, D


#### 5. Which of the following statements about positional encoding in the Transformer are true?  
A) ✗ The paper uses fixed sinusoidal encodings, not learned parameters (though learned embeddings were tested).  
B) ✓ Sinusoidal encodings help generalize to longer sequences.  
C) ✓ Positional encodings are added to embeddings to provide order information.  
D) ✗ No recurrent positional embeddings are used.  

**Correct:** B, C


#### 6. How does the Transformer’s decoder prevent attending to future tokens during training?  
A) ✓ Uses masking to set illegal attention weights to -∞ before softmax.  
B) ✗ It masks rather than limits the window size.  
C) ✗ No recurrent network is used in the decoder.  
D) ✓ Output embeddings are offset by one position to prevent peeking.  

**Correct:** A, D


#### 7. Which of the following are components of each encoder layer in the Transformer?  
A) ✓ Multi-head self-attention is a core sub-layer.  
B) ✓ Position-wise feed-forward network is the second sub-layer.  
C) ✗ Encoder-decoder attention is only in the decoder layers.  
D) ✓ Residual connections and layer normalization wrap each sub-layer.  

**Correct:** A, B, D


#### 8. What is the role of residual connections in the Transformer architecture?  
A) ✓ Facilitate gradient flow and stabilize training in deep networks.  
B) ✗ They do not reduce parameter count.  
C) ✗ Residuals do not combine attention heads.  
D) ✗ Normalization is a separate step, not residual connections.  

**Correct:** A


#### 9. Compared to convolutional sequence models like ConvS2S and ByteNet, self-attention layers:  
A) ✗ Convolutional models have path length growing with sequence length; self-attention does not.  
B) ✓ Self-attention has constant maximum path length \(O(1)\).  
C) ✓ Self-attention requires fewer operations to relate distant positions.  
D) ✗ Self-attention is more parallelizable than convolutional layers.  

**Correct:** B, C


#### 10. Why might additive attention outperform dot-product attention without scaling for large key dimensions?  
A) ✓ Additive attention uses a feed-forward network that can better model compatibility.  
B) ✗ Dot-product attention is faster, not more expensive.  
C) ✓ Unscaled dot products can become large, causing small gradients in softmax.  
D) ✗ Additive attention does not normalize keys and queries.  

**Correct:** A, C


#### 11. Which of the following describe the training regime used for the Transformer?  
A) ✓ Adam optimizer with warm-up learning rate schedule is used.  
B) ✓ Batches are grouped by approximate sequence length for efficiency.  
C) ✓ Label smoothing improves BLEU despite hurting perplexity.  
D) ✗ Training uses multiple GPUs, not single GPU exclusively.  

**Correct:** A, B, C


#### 12. In the Transformer, what is the dimensionality relationship between the number of attention heads (h), the model dimension (dmodel), and the key/value dimensions (dk, dv)?  
A) ✓ Each head’s key and value dimension is \(d_{model} / h\).  
B) ✗ Dimensions are not multiplied by the number of heads.  
C) ✗ Keys and values are not full model dimension per head.  
D) ✗ This ratio is inverted.  

**Correct:** A


#### 13. Which of the following are true about the feed-forward networks in the Transformer layers?  
A) ✓ Applied independently to each position.  
B) ✓ Two linear layers with ReLU activation in between.  
C) ✗ Parameters differ between layers; not shared.  
D) ✓ Equivalent to convolutions with kernel size 1.  

**Correct:** A, B, D


#### 14. How does the Transformer handle input and output token embeddings and the final softmax layer?  
A) ✗ Weight matrices are shared, not separate.  
B) ✓ Shares the same weight matrix for input embeddings, output embeddings, and pre-softmax linear layer.  
C) ✓ Embeddings are scaled by \(\sqrt{d_{model}}\) before adding positional encodings.  
D) ✗ Positional encodings are used in both encoder and decoder, not only decoder.  

**Correct:** B, C


#### 15. What is the main reason the Transformer can be trained faster than RNN-based models?  
A) ✗ Parameter count is not the main factor.  
B) ✓ Parallelization across all sequence positions during training.  
C) ✗ Transformer does not use convolutional layers.  
D) ✗ Number of training steps is not necessarily fewer.  

**Correct:** B


#### 16. Which of the following statements about the maximum path length in different layer types is correct?  
A) ✓ Recurrent layers have path length proportional to sequence length \(O(n)\).  
B) ✓ Self-attention layers have constant path length \(O(1)\).  
C) ✓ Dilated convolutions have path length \(O(\log_k n)\).  
D) ✗ Self-attention has shorter, not longer, path lengths than recurrent layers.  

**Correct:** A, B, C


#### 17. In the context of the Transformer, what is the significance of masking in the decoder’s self-attention?  
A) ✗ Masking here is for future tokens, not padding tokens.  
B) ✓ Prevents attending to future tokens, preserving autoregressive property.  
C) ✗ Masking is for correctness, not computational efficiency.  
D) ✗ Masking is applied during both training and inference.  

**Correct:** B


#### 18. Which of the following are challenges or limitations of self-attention that the Transformer addresses?  
A) ✓ Averaging attention can reduce effective resolution; multi-head attention counters this.  
B) ✗ Self-attention improves learning of long-range dependencies.  
C) ✓ Computational cost grows quadratically with sequence length, challenging very long sequences.  
D) ✓ Positional information must be explicitly added since no recurrence or convolution exists.  

**Correct:** A, C, D


#### 19. How does the Transformer generalize to tasks beyond machine translation, such as English constituency parsing?  
A) ✓ Uses the same architecture with minimal task-specific tuning.  
B) ✗ Does not add recurrent layers for other tasks.  
C) ✗ Number of attention heads is not necessarily increased.  
D) ✗ Does not require much larger datasets to perform well.  

**Correct:** A


#### 20. Which of the following statements about the learning rate schedule used in training the Transformer are true?  
A) ✓ Learning rate increases linearly during warm-up.  
B) ✓ After warm-up, learning rate decreases proportionally to inverse square root of step number.  
C) ✗ Learning rate is not constant throughout training.  
D) ✓ Warm-up steps are set to 4000 in the base model.  

**Correct:** A, B, D