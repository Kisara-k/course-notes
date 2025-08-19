## 5 Sequence Models

## Questions



#### 1. What are the main reasons standard feedforward neural networks are insufficient for sequence data?  
A) They cannot handle variable-length inputs and outputs.  
B) They do not share learned features across different positions in the sequence.  
C) They require labeled data for every time step.  
D) They inherently model temporal dependencies.

#### 2. In a vanilla RNN, what is the primary role of the hidden state at each time step?  
A) To store the entire input sequence.  
B) To capture information from previous time steps.  
C) To generate the final output only at the last time step.  
D) To reset the network’s memory after each input.

#### 3. Which of the following are true about Backpropagation Through Time (BPTT)?  
A) It unfolds the RNN across time steps to compute gradients.  
B) It can suffer from vanishing or exploding gradients.  
C) It updates weights only based on the last time step’s error.  
D) It is used to train feedforward networks on sequence data.

#### 4. Why do LSTM and GRU units improve upon vanilla RNNs?  
A) They use gating mechanisms to control information flow.  
B) They eliminate the need for backpropagation.  
C) They help mitigate vanishing gradient problems.  
D) They guarantee perfect long-term memory retention.

#### 5. Which statements about Bidirectional RNNs (BRNNs) are correct?  
A) They process sequences only in the forward direction.  
B) They combine information from past and future contexts.  
C) They require the entire sequence to be available before processing.  
D) They are unsuitable for real-time applications.

#### 6. In language modeling, what does the model primarily learn?  
A) To classify sentiment of sentences.  
B) To predict the next word given previous words.  
C) To translate sentences from one language to another.  
D) To generate sequences by sampling from learned probabilities.

#### 7. What are the challenges associated with training vanilla RNNs on long sequences?  
A) Exploding gradients cause unstable training.  
B) Vanishing gradients prevent learning long-range dependencies.  
C) They cannot process sequences longer than a fixed length.  
D) They require very large datasets to converge.

#### 8. Which of the following best describe word embeddings?  
A) Sparse one-hot vectors representing words.  
B) Dense vector representations capturing semantic similarity.  
C) Learned from large corpora or pre-trained and transferable.  
D) Fixed and cannot be fine-tuned for new tasks.

#### 9. How do word embeddings capture analogies such as “King - Man + Woman = Queen”?  
A) By encoding syntactic rules explicitly.  
B) Through linear relationships in the embedding space.  
C) By memorizing all possible word pairs during training.  
D) By clustering words with similar meanings together.

#### 10. What is the main purpose of debiasing word embeddings?  
A) To remove all semantic information from embeddings.  
B) To reduce gender, ethnicity, and other social biases learned from data.  
C) To improve model accuracy on sentiment classification.  
D) To ensure embeddings treat definitional and non-definitional words differently.

#### 11. In sequence-to-sequence models, why is beam search preferred over greedy search?  
A) Beam search explores multiple candidate sequences simultaneously.  
B) Beam search guarantees finding the globally optimal sequence.  
C) Greedy search can get stuck in locally optimal but suboptimal sequences.  
D) Beam search is always faster than greedy search.

#### 12. What is a key limitation of the basic encoder-decoder architecture without attention?  
A) It cannot handle variable-length input sequences.  
B) It compresses all input information into a fixed-size vector, limiting long sequence performance.  
C) It requires labeled data for every output token.  
D) It cannot be trained with backpropagation.

#### 13. How does the attention mechanism improve sequence-to-sequence models?  
A) By allowing the decoder to focus on different parts of the input sequence dynamically.  
B) By replacing the encoder entirely.  
C) By computing weighted sums of encoder outputs based on relevance.  
D) By eliminating the need for recurrent units.

#### 14. Which of the following are components of the Transformer architecture?  
A) Self-attention layers.  
B) Recurrent neural networks.  
C) Multi-head attention.  
D) Positional encoding.

#### 15. Why is positional encoding necessary in Transformers?  
A) Because Transformers process sequences in parallel and lack inherent order awareness.  
B) To replace the need for word embeddings.  
C) To encode the absolute position of words in the sequence.  
D) To improve the speed of training.

#### 16. In the context of speech recognition, what is the role of Connectionist Temporal Classification (CTC)?  
A) To align unsegmented audio inputs with text outputs.  
B) To segment audio into phonemes before recognition.  
C) To collapse repeated characters and blanks in output sequences.  
D) To generate audio from text.

#### 17. Which of the following statements about trigger word detection are true?  
A) It detects specific keywords in continuous audio streams.  
B) It requires large, complex models for real-time performance.  
C) It is used in devices like Amazon Alexa and Google Home.  
D) It uses the same architecture as machine translation models.

#### 18. What are the main differences between GRU and LSTM units?  
A) GRUs have fewer gates and are simpler than LSTMs.  
B) LSTMs have separate forget, input, and output gates.  
C) GRUs cannot handle long-term dependencies.  
D) LSTMs always outperform GRUs in every task.

#### 19. When evaluating machine translation, what does the BLEU score measure?  
A) The grammatical correctness of the output.  
B) The overlap of n-grams between the machine output and reference translations.  
C) The semantic similarity between sentences.  
D) The length ratio between output and reference.

#### 20. Which of the following are true about multi-head attention in Transformers?  
A) It allows the model to attend to information from different representation subspaces.  
B) It splits the input into multiple parts processed independently.  
C) It improves the model’s ability to capture diverse relationships in the data.  
D) It is only used during training, not inference.



<br>

## Answers



#### 1. What are the main reasons standard feedforward neural networks are insufficient for sequence data?  
A) ✓ They cannot handle variable-length inputs and outputs. Standard networks expect fixed-size inputs/outputs.  
B) ✓ They do not share learned features across different positions in the sequence. They treat each input independently.  
C) ✗ They do not necessarily require labeled data for every time step; this is task-dependent.  
D) ✗ They do not inherently model temporal dependencies; this is a limitation, not a reason they are sufficient.

**Correct:** A, B


#### 2. In a vanilla RNN, what is the primary role of the hidden state at each time step?  
A) ✗ It does not store the entire input sequence explicitly, only a summary.  
B) ✓ It captures information from previous time steps to maintain context.  
C) ✗ Outputs can be generated at each step, not only the last.  
D) ✗ The hidden state is updated, not reset after each input.

**Correct:** B


#### 3. Which of the following are true about Backpropagation Through Time (BPTT)?  
A) ✓ BPTT unfolds the RNN across time to compute gradients through all steps.  
B) ✓ It can suffer from vanishing or exploding gradients due to long sequences.  
C) ✗ It updates weights based on errors from all time steps, not just the last.  
D) ✗ BPTT is specific to RNNs, not feedforward networks.

**Correct:** A, B


#### 4. Why do LSTM and GRU units improve upon vanilla RNNs?  
A) ✓ They use gating mechanisms to control what information to keep or forget.  
B) ✗ They still require backpropagation for training.  
C) ✓ They help mitigate vanishing gradient problems by better gradient flow.  
D) ✗ They do not guarantee perfect long-term memory, only improve it.

**Correct:** A, C


#### 5. Which statements about Bidirectional RNNs (BRNNs) are correct?  
A) ✗ BRNNs process sequences in both forward and backward directions.  
B) ✓ They combine past and future context for better understanding.  
C) ✓ They require the entire sequence before processing since backward pass depends on future.  
D) ✗ They can be adapted for real-time with some delay, but generally less suitable.

**Correct:** B, C


#### 6. In language modeling, what does the model primarily learn?  
A) ✗ Sentiment classification is a different task.  
B) ✓ Predicting the next word given previous words is the core task.  
C) ✗ Machine translation is related but distinct.  
D) ✓ Generating sequences by sampling from learned probabilities is a use of language models.

**Correct:** B, D


#### 7. What are the challenges associated with training vanilla RNNs on long sequences?  
A) ✓ Exploding gradients cause unstable training if not controlled.  
B) ✓ Vanishing gradients prevent learning dependencies far back in time.  
C) ✗ They can process sequences of arbitrary length, but performance degrades.  
D) ✗ Large datasets help but do not solve gradient issues.

**Correct:** A, B


#### 8. Which of the following best describe word embeddings?  
A) ✗ One-hot vectors are sparse and do not capture similarity.  
B) ✓ Dense vectors capture semantic similarity between words.  
C) ✓ They are learned from large corpora and can be transferred to new tasks.  
D) ✗ Embeddings can be fine-tuned for specific tasks.

**Correct:** B, C


#### 9. How do word embeddings capture analogies such as “King - Man + Woman = Queen”?  
A) ✗ They do not encode explicit syntactic rules.  
B) ✓ Linear relationships in vector space allow analogies.  
C) ✗ They do not memorize all pairs but learn generalizable patterns.  
D) ✓ Words with similar meanings cluster together, supporting analogies.

**Correct:** B, D


#### 10. What is the main purpose of debiasing word embeddings?  
A) ✗ Debiasing does not remove semantic information, only unwanted bias.  
B) ✓ It reduces social biases like gender or ethnicity present in training data.  
C) ✗ Debiasing is about fairness, not directly improving sentiment accuracy.  
D) ✓ It treats definitional words differently from non-definitional to preserve meaning.

**Correct:** B, D


#### 11. In sequence-to-sequence models, why is beam search preferred over greedy search?  
A) ✓ Beam search keeps multiple hypotheses, improving chances of better sequences.  
B) ✗ Beam search does not guarantee global optimality, only approximates it.  
C) ✓ Greedy search can get stuck in locally optimal but suboptimal sequences.  
D) ✗ Beam search is generally slower than greedy search due to multiple candidates.

**Correct:** A, C


#### 12. What is a key limitation of the basic encoder-decoder architecture without attention?  
A) ✗ It can handle variable-length inputs and outputs.  
B) ✓ Compressing all input into a fixed vector limits performance on long sequences.  
C) ✗ It does not require labeled data for every output token, depends on task.  
D) ✗ It can be trained with backpropagation.

**Correct:** B


#### 13. How does the attention mechanism improve sequence-to-sequence models?  
A) ✓ It allows the decoder to dynamically focus on relevant input parts.  
B) ✗ It does not replace the encoder but complements it.  
C) ✓ It computes weighted sums of encoder outputs based on relevance scores.  
D) ✗ It does not eliminate the need for recurrent units, though some models do.

**Correct:** A, C


#### 14. Which of the following are components of the Transformer architecture?  
A) ✓ Self-attention layers are core to Transformers.  
B) ✗ Transformers do not use recurrent neural networks.  
C) ✓ Multi-head attention allows attending to multiple aspects simultaneously.  
D) ✓ Positional encoding adds order information.

**Correct:** A, C, D


#### 15. Why is positional encoding necessary in Transformers?  
A) ✓ Transformers process sequences in parallel and lack inherent order awareness.  
B) ✗ Positional encoding supplements, not replaces, word embeddings.  
C) ✓ It encodes absolute or relative position of words in the sequence.  
D) ✗ It does not directly improve training speed but model effectiveness.

**Correct:** A, C


#### 16. In the context of speech recognition, what is the role of Connectionist Temporal Classification (CTC)?  
A) ✓ Aligns unsegmented audio inputs with text outputs without explicit segmentation.  
B) ✗ It does not segment audio into phonemes explicitly.  
C) ✓ Collapses repeated characters and blanks to produce final transcription.  
D) ✗ CTC is for recognition, not generation of audio.

**Correct:** A, C


#### 17. Which of the following statements about trigger word detection are true?  
A) ✓ Detects specific keywords like “Alexa” in continuous audio.  
B) ✗ Models are often lightweight for real-time performance, not large/complex.  
C) ✓ Used in commercial voice assistants like Alexa, Siri, Google Home.  
D) ✗ Uses specialized architectures, not the same as machine translation models.

**Correct:** A, C


#### 18. What are the main differences between GRU and LSTM units?  
A) ✓ GRUs have fewer gates and simpler structure than LSTMs.  
B) ✓ LSTMs have separate forget, input, and output gates.  
C) ✗ GRUs can handle long-term dependencies, though sometimes less effectively.  
D) ✗ Neither always outperforms the other; performance depends on task and data.

**Correct:** A, B


#### 19. When evaluating machine translation, what does the BLEU score measure?  
A) ✗ It does not directly measure grammatical correctness.  
B) ✓ Measures n-gram overlap precision between output and references.  
C) ✗ Semantic similarity is not directly measured by BLEU.  
D) ✓ Considers length ratio to penalize overly short or long outputs.

**Correct:** B, D


#### 20. Which of the following are true about multi-head attention in Transformers?  
A) ✓ Allows attending to different representation subspaces simultaneously.  
B) ✗ Input is not split arbitrarily but projected into multiple heads.  
C) ✓ Improves ability to capture diverse relationships in data.  
D) ✗ Used during both training and inference.

**Correct:** A, C

