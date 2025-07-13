## 5 Sequence Models



### Key Points



#### 1. 🔄 Recurrent Neural Networks (RNNs)  
- RNNs process sequences by maintaining a hidden state that updates at each time step based on current input and previous hidden state.  
- Standard neural networks cannot handle variable-length sequences or share features across positions in a sequence.  
- Backpropagation Through Time (BPTT) is used to train RNNs by propagating errors through all time steps.

#### 2. 🧠 Gated RNN Architectures (GRU and LSTM)  
- GRUs use update and reset gates to control information flow, improving training stability.  
- LSTMs have three gates: forget, input (update), and output gates, enabling learning of long-term dependencies.  
- Both GRU and LSTM help mitigate vanishing and exploding gradient problems in RNNs.

#### 3. 🌐 Word Embeddings  
- Word embeddings map words to dense vectors capturing semantic similarity (e.g., King - Man + Woman ≈ Queen).  
- Word2Vec (skip-gram) and GloVe are popular methods for learning word embeddings from large corpora.  
- Embeddings can be transferred to new tasks and fine-tuned with smaller datasets.  
- Word embeddings can encode biases (gender, ethnicity), which can be identified and reduced via debiasing techniques.

#### 4. 🔄 Sequence-to-Sequence (Seq2Seq) Models  
- Seq2Seq models use an encoder-decoder architecture to map input sequences to output sequences (e.g., machine translation).  
- Beam search is used during decoding to keep multiple candidate sequences, improving output quality over greedy search.  
- Length normalization in beam search prevents bias toward shorter sequences.  
- Error analysis distinguishes errors caused by beam search from those caused by the model itself.

#### 5. 🎯 Attention Mechanism  
- Attention allows the decoder to focus on different parts of the input sequence dynamically, improving handling of long sequences.  
- Attention computes alignment scores between decoder states and encoder outputs, producing weighted context vectors.  
- Attention improves translation quality and provides interpretability by showing which input words are focused on.

#### 6. 🗣️ Speech Recognition and Trigger Word Detection  
- Connectionist Temporal Classification (CTC) loss aligns unsegmented audio with text by collapsing repeated characters and blanks.  
- Trigger word detection identifies specific keywords (e.g., “Alexa”) in continuous audio streams using lightweight RNNs.

#### 7. ⚡ Transformers  
- Transformers replace recurrence with self-attention, allowing parallel processing of sequences.  
- Multi-head attention captures different types of relationships by computing attention multiple times in parallel.  
- Transformers use positional encoding to incorporate word order information.  
- Transformers have become state-of-the-art in machine translation and other sequence tasks due to efficiency and ability to model long-range dependencies.

#### 8. 📝 Language Modeling  
- Language models predict the next word in a sequence given previous words, trained on large text corpora.  
- RNN-based language models generate text by sampling one word at a time, feeding outputs back as inputs.  
- Vanishing gradients limit RNNs’ ability to learn long-range dependencies, addressed by gated units like LSTM and GRU.



<br>

## Study Notes



### Study Notes: Sequence Models and NLP Fundamentals


### 1. 🎵 Why Sequence Models? Understanding the Need for Sequential Data Processing

Sequence models are designed to handle data where the order of elements matters. Unlike traditional machine learning models that treat inputs as independent, sequence models capture dependencies and relationships across time or position in data.

#### What is Sequence Data?

Sequence data appears in many real-world applications, such as:

- **Music generation:** Notes played in a sequence create melodies.
- **Speech recognition:** Audio signals unfold over time.
- **Text processing:** Sentences like “The quick brown fox jumped over the lazy dog” depend on word order.
- **Sentiment classification:** Understanding if “There is nothing to like in this movie” expresses positive or negative sentiment.
- **DNA sequence analysis:** Genetic information is a sequence of nucleotides (e.g., AGCCCCTGTGAGGAACTAG).
- **Machine translation:** Translating “Voulez-vous chanter avec moi?” to “Do you want to sing with me?”
- **Video activity recognition:** Recognizing actions like running from a sequence of video frames.
- **Named entity recognition:** Identifying names in “Yesterday, Harry Potter met Hermione Granger.”

#### Why Not Use Standard Neural Networks?

Standard feedforward networks expect fixed-size inputs and outputs and do not share learned features across positions in a sequence. But sequences vary in length, and the meaning of a word depends on its context. Sequence models, especially Recurrent Neural Networks (RNNs), address these challenges by processing data step-by-step and maintaining a memory of previous inputs.


### 2. 🔄 Recurrent Neural Networks (RNNs): The Backbone of Sequence Modeling

#### What is an RNN?

An RNN is a type of neural network designed to process sequences by maintaining a hidden state that captures information from previous time steps. At each step, the RNN takes an input (e.g., a word or character) and updates its hidden state, which influences future predictions.

#### Notation and Example

Consider the sentence:  
*“Harry Potter and Hermione Granger invented a new spell.”*

Each word is represented as a vector (often an integer index or embedding). For example:  
- Harry = 4075  
- Potter = 6830  
- And = 367  
- Invented = 4700  
- New = 5976  
- Spell = 8376

The RNN processes these words one at a time, updating its hidden state to remember context.

#### Why Not Use a Standard Network?

- **Variable length:** Sentences can be short or long.
- **Context sharing:** Features learned at one position should help understand other positions.
- **Sequential dependency:** The meaning of a word depends on previous words.

#### Forward Propagation in RNNs

At each time step \( t \), the RNN updates its hidden state \( h_t \) based on the current input \( x_t \) and the previous hidden state \( h_{t-1} \):

\[
h_t = \tanh(W_{xh} x_t + W_{hh} h_{t-1} + b_h)
\]

The output \( y_t \) can be computed from \( h_t \) using another weight matrix.

#### Backpropagation Through Time (BPTT)

Training RNNs involves backpropagating errors through all time steps, called Backpropagation Through Time. This allows the network to learn how earlier inputs affect later outputs.


### 3. 🧠 Advanced RNN Architectures: GRU and LSTM

#### The Vanishing and Exploding Gradient Problem

Standard RNNs struggle to learn long-range dependencies because gradients can vanish (become too small) or explode (become too large) during training.

#### Gated Recurrent Unit (GRU)

GRUs introduce gating mechanisms to control information flow, helping the network remember or forget information as needed. This makes training more stable and effective.

- **Update gate:** Decides how much past information to keep.
- **Reset gate:** Controls how much past information to forget.

#### Long Short-Term Memory (LSTM)

LSTMs are more complex units with three gates:

- **Forget gate:** Decides what information to discard.
- **Input (update) gate:** Decides what new information to add.
- **Output gate:** Controls what information to output.

LSTMs can capture long-term dependencies better than vanilla RNNs.


### 4. 🔄 Bidirectional and Deep RNNs

#### Bidirectional RNNs (BRNNs)

Sometimes, understanding a word depends on both past and future context. BRNNs process sequences in both forward and backward directions, combining information from the past and future.

Example:  
- Forward: “He said, ‘Teddy Roosevelt was a great President.’”  
- Backward: “He said, ‘Teddy bears are on sale!’”

#### Deep RNNs

Stacking multiple RNN layers (deep RNNs) allows the model to learn more complex patterns by processing the sequence through several transformations.


### 5. 📝 Language Modeling and Sequence Generation

#### What is Language Modeling?

Language modeling is the task of predicting the next word in a sequence given the previous words. For example, given “Cats average 15 hours of sleep a day,” the model learns to predict the next word or the end of the sentence.

#### Training a Language Model with RNNs

- Use a large corpus of text.
- At each step, predict the next word.
- Calculate loss using cross-entropy between predicted and actual next word.
- Train using backpropagation through time.

#### Sampling Novel Sequences

After training, the RNN can generate new text by sampling one word at a time, feeding the output back as input for the next step. This can produce coherent sentences or even creative text like news or Shakespearean style.


### 6. ⚠️ Challenges: Vanishing and Exploding Gradients

- **Vanishing gradients:** Gradients become too small, preventing learning of long-range dependencies.
- **Exploding gradients:** Gradients become too large, causing unstable training.

Solutions include gradient clipping, using gated units like LSTM/GRU, and careful initialization.


### 7. 🌐 Word Embeddings: Representing Words as Vectors

#### Why Word Embeddings?

One-hot vectors (e.g., [0,0,1,0,...]) are sparse and do not capture semantic similarity. Word embeddings map words to dense vectors in a continuous space where similar words are close together.

#### Examples of Word Embeddings

- **Apple** and **Orange** are close because both are fruits.
- **King** and **Queen** are close and relate to gender and royalty.
- Visualizations like t-SNE show clusters of related words.

#### Learning Word Embeddings

- Train on large corpora (billions of words).
- Use models like Word2Vec (skip-gram, CBOW) or GloVe.
- Embeddings can be transferred to new tasks and fine-tuned.

#### Properties of Word Embeddings

- Capture analogies:  
  *King - Man + Woman ≈ Queen*  
  *Paris - France + Italy ≈ Rome*
- Reflect semantic and syntactic relationships.

#### Debiasing Word Embeddings

Embeddings can inherit biases (gender, ethnicity) from training data. Techniques to reduce bias include:

1. Identifying bias directions.
2. Neutralizing non-definitional words.
3. Equalizing pairs (e.g., man-woman).


### 8. 🔄 Sequence-to-Sequence Models: Translating and Generating Sequences

#### What is a Sequence-to-Sequence (Seq2Seq) Model?

Seq2Seq models map an input sequence (e.g., French sentence) to an output sequence (e.g., English translation). They typically use an encoder-decoder architecture:

- **Encoder:** Processes input sequence into a fixed-size context vector.
- **Decoder:** Generates output sequence from the context vector.

#### Applications

- Machine translation: “Jane visite l'Afrique en septembre” → “Jane is visiting Africa in September.”
- Image captioning: Generate text descriptions from images.
- Speech recognition.

#### Beam Search for Decoding

Instead of greedily picking the most likely next word, beam search keeps multiple candidate sequences (beam width \( B \)) to find better overall translations.

- Balances speed and quality.
- Uses length normalization to avoid bias toward shorter sequences.

#### Error Analysis

Errors can come from:

- Beam search choosing suboptimal sequences.
- The model predicting incorrect probabilities.


### 9. 🎯 Attention Mechanism: Improving Seq2Seq Models

#### Why Attention?

Long sequences cause problems because the encoder compresses all information into a single vector. Attention allows the decoder to focus on different parts of the input sequence dynamically.

#### How Attention Works

- Compute alignment scores between decoder state and each encoder output.
- Normalize scores to get attention weights.
- Compute a weighted sum of encoder outputs as context for each decoding step.

#### Benefits

- Better handling of long sentences.
- Improved translation quality.
- Interpretability: visualize which input words the model attends to.


### 10. 🗣️ Speech Recognition and Trigger Word Detection

#### Speech Recognition

Convert audio waveforms into text transcripts. Challenges include variable length, noise, and timing.

- Use RNNs with attention or Connectionist Temporal Classification (CTC) loss.
- CTC aligns unsegmented audio with text by collapsing repeated characters and blanks.

#### Trigger Word Detection

Detect specific keywords like “Alexa” or “Hey Siri” in continuous audio streams.

- Requires fast, accurate detection.
- Often uses specialized lightweight RNNs.


### 11. ⚡ Transformers: The New Standard in Sequence Modeling

#### Motivation

RNNs process sequences sequentially, which limits parallelization and struggles with very long dependencies.

#### Transformer Architecture

- Uses **self-attention** to relate all words in a sequence simultaneously.
- Employs **multi-head attention** to capture different types of relationships.
- Consists of an **encoder** and **decoder** with feed-forward layers and normalization.

#### Key Concepts

- **Query (Q), Key (K), Value (V):** Vectors used to compute attention scores.
- **Self-attention:** Each word attends to all others in the sequence.
- **Positional encoding:** Adds information about word order since transformers lack recurrence.

#### Advantages

- Highly parallelizable.
- Better at capturing long-range dependencies.
- State-of-the-art in machine translation, text generation, and more.


### Summary

Sequence models are essential for processing data where order matters, such as text, speech, and DNA. RNNs, especially with gated units like LSTM and GRU, enable learning from sequences but have limitations with long dependencies. Word embeddings provide meaningful vector representations of words, improving NLP tasks. Sequence-to-sequence models with attention revolutionized translation and generation tasks. Finally, transformers, based on self-attention, have become the dominant architecture for sequence modeling due to their efficiency and power.



<br>

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

