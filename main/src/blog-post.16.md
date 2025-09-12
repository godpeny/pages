# Transformers
## Motivation
Seq2seq model is fundamentally RNN based sequential model which means the model has to ingest the input one token(word) at a time. 
So each unit was like a bottleneck to the flow of information. Simply speaking, in order to compute the final unit (of encoder, for example), you first have to compute the outputs of all of the units that come before. Plus, with GRU/LSTM applied to cope with vanishing gradient problem, the complexity of model increased. 

Transformers architecture allow you to to run a lot more of computation of units for an entire sequence in parallel. In fact in ingests an entire input sentence all at the same time rather than processing then one at a time. Also, transformer have the advantage of having no recurrent units, therefore requiring less training time than earlier recurrent neural architectures (RNNs) such as long short-term memory (LSTM), because there are less parameters to train. 

## Intuition
The major innovation of transformers architecture is combining the use of attention-based representation and convolutional neural network (CNN) style of processing.
<img src="images/blog16_transformers_intuition.png" alt="Transformer Intuition" width="400"/>  
As you can see from the above spicture, unlike RNN based seq2seq models, attransformers take input of a lot of words(pixels) and compute representation for them in parallel.  
Also note that there are two key ideas in transformers, which are <b> Self-Attention </b> and 
<b> Multi-Head Attention </b>.

## End-to-End Memory

## Architecture
<img src="images/blog16_transformers_architecture.png" alt="Transformer Intuition" width="400"/>  

## Attention(recap)
### Model Architecture
The attention model's architecture consists of a bidirectional RNN as an encoder and a decoder that emulates searching through a source sentence during decoding a translation.
<img src="images/blog4_attention_architecture.png" alt="Model architecture" width="200"/>  

Above image shows the new model trying to generate the $t$-th target
word $y_t$ given a source sentence $(x_1, x_2, \cdots, x_T)$.

#### Encoder
An encoder is the proposed scheme, we would like the annotation  $(h_1, \cdots, h_{T_x})$
of each word to summarize not only the preceding words, but also the following words. Hence,
we propose to use a bidirectional RNN (BiRNN).  
An annotation for each word $x_j$ by concatenating the forward hidden state
$\vec{h}_j$ and the backward one $ \overleftarrow{h}_j$. 
$$
h_j = \begin{bmatrix} \overrightarrow{h}_j^{\top} \; ; \; \overleftarrow{h}_j^{\top} \end{bmatrix}^{\top}
$$
In this way, the annotation $h_j$ contains the summaries of both the preceding words and the following words.

##### Deepen Explanation on Encoder
The forward states of the bidirectional recurrent neural network (BiRNN) are computed,
$$
\overrightarrow{h}_i =
\begin{cases}
(1 - \overrightarrow{z}_i) \circ \overrightarrow{h}_{i-1} + \overrightarrow{z}_i \circ \overrightarrow{\tilde{h}}_i, & \text{if } i > 0 \\
0, & \text{if } i = 0
\end{cases}
$$
Where,
$$
\overrightarrow{\tilde{h}}_i = \tanh \left( \overrightarrow{W} \, \overline{E} x_i + \overrightarrow{U} \left[ \overrightarrow{r}_i \circ \overrightarrow{h}_{i-1} \right] \right) \\[5pt]
\overrightarrow{z}_i = \sigma \left( \overrightarrow{W}_z \, \overline{E} x_i + \overrightarrow{U}_z \, \overrightarrow{h}_{i-1} \right) \\[5pt]
\overrightarrow{r}_i = \sigma \left( \overrightarrow{W}_r \, \overline{E} x_i + \overrightarrow{U}_r \, \overrightarrow{h}_{i-1} \right)
$$
$\overline{E} \in \mathbb{R}^{m \times K_x}$ is the word embedding matrix and $\overrightarrow{W}, \; \overrightarrow{W}_z, \; \overrightarrow{W}_r \in \mathbb{R}^{n \times m}, \overrightarrow{U}, \; \overrightarrow{U}_z, \; \overrightarrow{U}_r \in \mathbb{R}^{n \times n}$ are weight matrices. $m,n$ are the word embedding dimensionality and the number of hidden units, respectively. $\sigma$ is a logistic sigmoid function.

The backward states are computed similarly. So we concatenate the forward and backward states to to obtain the annotations $(h_1, h_2, \cdots, h_{Tx} )$, where, 
$$
h_i = 
\begin{bmatrix}
\overrightarrow{h}_i \\
\overleftarrow{h}_i
\end{bmatrix}
$$

#### Decoder (+ Alignment Model)
The decoder is trained to predict the next word $y_{t^{'}}$ given the context vector $c$ and all the previously predicted words $\{y_1, \cdots, y_{t^{'}-1}\}$. The difference between previous model is that the probability is conditioned on a distinct context vector $c_i$ for each target word $y_i$, instead of using a single fixed-length vector of whole input sentence.
$$
p(y_i \mid y_1, \ldots, y_{i-1}, \mathbf{x}) 
= g(y_{i-1}, s_i, c_i), \\[5pt]
s_i = f(s_{i-1}, y_{i-1}, c_i)
$$
The $p$ represents the conditional probability over the translation $y$ and $s_i$ is an RNN hidden state for time $i$. See the moddel architecture image for better understanding.

$$
c_i = \sum_{j=1}^{T_x} \alpha_{ij} h_j, \quad
\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k=1}^{T_x} \exp(e_{ik})}, \quad
e_{ij} = a(s_{i-1}, h_j)
$$
(annotations $h$ is explained in previous "Encoder" part.)  

Next, the context vector $c_i$ depends on a sequence of annotations $(h_1, \cdots, h_{T_x})$ to which an encoder maps the input sentence. Each annotation $h_i$ contains information about the whole input sequence with a strong focus on the parts surrounding the $i$-th word of the input sequence. So taking a weighted sum of all the annotations can be interpreted as computing an expected annotation, where the expectation is over possible alignments.

$\alpha$ is a weight. So $\alpha_{ij}$ be a probability that the target word $y_i$ is aligned to, or translated from, a source word $x_j$. Then, the $i$-th context vector $c_i$ is the expected annotation over all the annotations with probabilities $\alpha_{ij}$.

$e_{ij}$ is an alignment model which scores how well the inputs around position $j$ and the output at position $i$ match. The score is based on the RNN hidden state $s_{i-1}$ (just before emitting $y_i$) and the $j$-th annotation $h_j$. It provides an intuitive way to inspect the (soft-)alignment between the words in a generated translation and those in a source sentence.

So, The probability $\alpha_{ij}$, or its associated energy $e_{ij}$, reflects the importance of the annotation $h_j$ with respect to the previous hidden state $s_{i-1}$ in deciding the next state $s_i$ and generating $y_i$. Intuitively, this implements a mechanism of attention in the decoder. The decoder decides parts of the source sentence to pay attention to. By letting the decoder have an attention mechanism, we relieve the
encoder from the burden of having to encode all information in the source sentence into a fixed length vector.

##### Deepen Explanation on Decoder
Let's see how the hidden state $s_i$ of the decoder described above was actually implemented.
$$
s_i = f(s_{i-1}, y_{i-1}, c_i) = (1 - z_i) \circ s_{i-1} + z_i \circ \tilde{s}_i,
$$
Where,
$$
\tilde{s}_i = \tanh \big( W E y_{i-1} + U \,[ r_i \circ s_{i-1}] + C c_i \big), \\[5pt]
z_i = \sigma \big( W_z E y_{i-1} + U_z s_{i-1} + C_z c_i \big), \\[5pt]
r_i = \sigma \big( W_r E y_{i-1} + U_r s_{i-1} + C_r c_i \big)
$$
Where $W, W_z, W_r \in \mathbb{R}^{n \times m}, \quad U, U_z, U_r \in \mathbb{R}^{n \times n}, C, C_z, C_r \in \mathbb{R}^{n \times 2n}$ are weights and $E$ is word embedding matrix for target language. Also $m, n$ are the word embedding dimensionality and the number of hidden units, respectively.

Next, let's see how the context vector $c_i$ is actually implemented. More precisely, how the allignment model is.
$$
c_i = \sum_{j=1}^{T_x} \alpha_{ij} h_j, \\[5pt]
\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k=1}^{T_x} \exp(e_{ik})}, \\[5pt]
e_{ij} = a(s_{i-1}, h_j) =  v_a^{\top} \tanh \left( W_a s_{i-1} + U_a h_j \right),
$$
Where $h_j$ is the $j$-th annotation in the source sentence. $v_a^{\top} \in \mathbb{R}^{n'}, W_a \in \mathbb{R}^{n' \times n}, U_a \in \mathbb{R}^{n' \times 2n}$.
Lastly, let's see the probability of a target word $y_i$ we described before.
$$
p(y_i \mid s_{i-1}, y_{i-1}, c_i) \;\propto\; \exp \!\big( y_i^\top W_o t_i \big) \\[5pt] 
\rightarrow p(y_i \mid s_{i-1}, y_{i-1}, c_i) = g(y_{i-1}, s_i, c_i) = \frac{\exp\!\big(y_i^\top W_o t_i\big)} {\sum_{k=1}^{T_y} \exp\!\big(y_k^\top W_o t_i\big)}, \\[5pt]
t_i = \Big[ \max \{ \tilde{t}_{i,2j-1}, \tilde{t}_{i,2j} \} \Big]_{j=1,\ldots,l}^{\top}, \quad
\tilde{t}_i = U_o s_{i-1} + V_o E y_{i-1} + C_o c_i.
$$
Where, $W_o \in \mathbb{R}^{K_y \times l}, \quad  U_o \in \mathbb{R}^{2l \times n}, \quad  V_o \in \mathbb{R}^{2l \times m}, \quad C_o \in \mathbb{R}^{2l \times 2n}$ are weight matrices. Note that first double the dimension with $\tilde{t}_i \in \mathbb{R}^{2l}$, then reduce it back to $\ell$ via maxout pooling, which picks the stronger (max) activation from each pair.

## Self-Attention (Intra-attention)
Self-attention is an attention mechanism relating different positions of a single sequence in order to compute a representation of the sequence. 

The Self-Attention model maintains two sets of vectors stored in a hidden state tape used to interact with the environment (e.g., computing attention), and a memory tape used to represent what is actually stored in memory. For comparison, LSTMs maintain a hidden vector and a memory vector; memory networks have a set of key vectors and a set of value vectors.  

Our solution is to modify the standard LSTM structure by replacing the memory cell with a memory network.
This design enables the LSTM to reason about relations between tokens with a neural attention layer and then perform non-Markov state updates.which means update its states using information from the whole history, not just the last hidden state.

## Attention vs Self-Attention
<b>Attention</b>  
Across different sequences (e.g.,encoder–decoder). So the encoder hidden states from the source sentence is one sequence(Keys/Values) and the decoder state while generating is another sequence(Query).

<b>Self-Attention</b>  
In self-attention (e.g. Transformer encoder), queries, keys, and values all come from the same sequence.

In summary, 
- (general) Attention
  - cross-sequence (decoder ↔ encoder).
- Self-attention
  - within-sequence (tokens ↔ tokens in the same sentence).


### Memory Network

## Multi-Head Attention
## Positional Encoding
## Transformer Network
