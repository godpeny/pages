# Sequence Model and Attention Mechanism
## Preliminaries
### Statistical Machine Translation (SMT)
SMT is a type of machine translation that uses statistical models to translate text from one language to another. Unlike traditional rule-based systems, SMT relies on large bilingual text corpora to build probabilistic models that determine the likelihood of a sentence in the target language given a sentence in the source language.  
This approach marked a significant shift in natural language processing (NLP) and more advanced machine translation technologies.

## RNN Encoder-Decoder Model
RNN Encoder–Decoder consists of two recurrent neural networks (RNN) that act as an encoder and a decoder pair. The encoder
maps a variable-length source sequence to a fixed-length vector, and the decoder maps the vector representation back to a variable-length target sequence. The two networks are trained jointly to maximize the conditional probability of the target sequence given a source sequence.
![alt text](images/blog4_rnn_encoder_decoder_arch.png)  
Encoder-Decoder architecture that learns to encode a variable-length sequence into a fixed-length vector representation and to decode a given fixed-length vector representation back into a variable-length sequence.
From a probabilistic perspective, this new model
is a general method to learn the conditional distribution
over a variable-length sequence conditioned on another variable-length sequence. For example,
$$
p(y_1, \ldots, y_{T'} \mid x_1, \ldots, x_T), \\[5pt]
$$
Where lengths $T$ and $T'$ may differ.  

<b> Encoder </b>  
The encoder is an RNN that reads each symbol
of an input sequence $x$ sequentially. As it reads
each symbol, the hidden state of the RNN changes
according to the equaton below. 
$$
\mathbf{h}_{\langle t \rangle} = f \big( \mathbf{h}_{\langle t-1 \rangle}, x_t \big)
$$
Where $t$ represents time step, $\mathbf{h}_{\langle t \rangle}$ is hidden state at time step $t$, $x = (x_1,  x_2, \cdots x_t)$ is input sequence and $f$ is a non-linear activation function.  
After reading the end of the sequence, the hidden state of the RNN is a summary $c$ of the whole input sequence.

<b> Decoder </b>  
The decoder of the model is another RNN which is trained to generate the output sequence by predicting the next symbol $y_{t}$ given the hidden state $\mathbf{h}_{\langle t \rangle}$. So the hidden state of the decoder and the output at time $t$ are computed by,
$$
\mathbf{h}_{\langle t \rangle} = f \big( \mathbf{h}_{\langle t-1 \rangle}, y_{t-1}, c \big), \\[5pt]
P(y_t \mid y_{t-1}, y_{t-2}, \ldots, y_1, c) 
= g \big( \mathbf{h}_{\langle t \rangle}, y_{t-1}, c \big)
$$
You can see that  both $y_{t}$ and $\mathbf{h}_{\langle t \rangle}$ are also conditioned $y_{t-1}$ and on the summary $c$ of the input sequence. 

<b> Objective Function </b>  
The objective function of the model is as below.proposed
model architecture. The RNN Encoder–Decoder are jointly trained to maximize this conditional log-likelihood. 
$$
\max_{\theta} \; \frac{1}{N} \sum_{n=1}^{N} \log p_\theta(y_n \mid x_n)
$$
Where $\theta$ is the set of the model parameters and
each $(x_n, y_n)$ is an (input sequence, output sequence)
pair from the training set.  

Note that this objective function is equal to that of RNN model. Recall that,
$$
\begin{aligned}
\mathcal{L}(\hat{y}^{\langle t \rangle}, y^{\langle t \rangle}) &= - \sum_i y_i^{\langle t \rangle} \log \hat{y}_i^{\langle t \rangle} \\
\mathcal{L} &= \sum_t \mathcal{L}^{\langle t \rangle}(\hat{y}^{\langle t \rangle}, y^{\langle t \rangle})
\end{aligned}
$$
Now let's see how they are equal.
$$
p_\theta(y_n \mid x_n) 
= \prod_{t=1}^{T_n} p_\theta\!\big(y_n^{\langle t \rangle} \mid y_n^{\langle <t \rangle}, x_n \big), \\[5pt]
\log p_\theta(y_n \mid x_n) 
= \sum_{t=1}^{T_n} \log p_\theta\!\big(y_n^{\langle t \rangle} \mid y_n^{\langle <t \rangle}, x_n \big)
$$
Since the probability of the true word is just the corresponding predicted probability, 
$$
p_\theta\!\big(y_n^{\langle t \rangle} \mid y_n^{\langle <t \rangle}, x_n \big) = \hat{y}_{n,k}^{\langle t \rangle}, \\[5pt]
\log p_\theta \big(y_n^{\langle t \rangle} \mid y_n^{\langle <t \rangle}, x_n  \big) = \log \hat{y}_{n,k}^{\langle t \rangle}
$$
You can interprete it as the probability of the true word in the $n$-th training example at timestep $t$ is one of the entries in the $K$-dimensional softmax output vector.  
Now, using the fact that $y_n^{\langle t \rangle}$ is one-hot vector that only one component is $1$ (the true class) and the rest are $0$, 
$$
\log p_\theta \big(y_n^{\langle t \rangle} \mid y_n^{\langle <t \rangle}, x_n  \big) = \log \hat{y}_{n,k}^{\langle t \rangle} \\[3pt]
= \sum_{i=1}^K y_{n,i}^{\langle t \rangle} \log \hat{y}_{n,i}^{\langle t \rangle}
$$
Therefore, 
$$
\log p_\theta(y_n \mid x_n) = \sum_{t=1}^{T_n} \log p_\theta\!\big(y_n^{\langle t \rangle} \mid y_n^{\langle <t \rangle}, x_n \big) \\[3pt]
= \sum_{t=1}^{T_n} \sum_{i=1}^{K} y_{n,i}^{\langle t \rangle} \log \hat{y}_{n,i}^{\langle t \rangle}
$$

## Sequence to Sequence Model
The Sequence-to-Sequence (Seq2Seq) model is a type of neural network architecture widely used in machine learning particularly in tasks that involve translating one sequence of data into another. It takes an input sequence, processes it and generates an output sequence. The Seq2Seq model has made significant contributions to areas such as natural language processing (NLP), machine translation and speech recognition.
![alt text](images/blog4_seq2seq.png)  

It uses RNN Encoder-Decoer architecture for general sequence learning which is mapping the input sequence to a fixed-sized vector using one RNN, and then to map the vector to the target sequence with another RNN. While it would be difficult to train the RNNs due to the resulting long term dependencies, the LSTM’s ability to successfully learn on data with long range temporal dependencies makes it a natural choice for this application due to the considerable time lag between the inputs and their corresponding outputs.  

Note that The goal of the LSTM is to estimate the conditional probability $p(y_1, \cdots, y_{T′} |x_1, \cdots , x_T)$ where $(x_1, \cdots , x_T )$ is an input sequence and $y_1, \cdots, y_{T′}$ is its corresponding output sequence whose length $T′$ may differ from $T$.The LSTM computes this conditional probability by first obtaining the fixed dimensional representation $v$ of the input sequence $(x_1, \cdots , x_T )$ given by the last hidden state of the LSTM, and then computing the probability of $y_1, \cdots, y_{T′}$ with a standard LSTM-LM(Language Model) formulation whose initial hidden state is set to the representation $v$ of $x_1, \cdots , x_T$: 
$$
p(y_1, \ldots, y_{T'} \mid x_1, \ldots, x_T) 
= \prod_{t=1}^{T'} p(y_t \mid v, y_1, \ldots, y_{t-1})
$$
In this equation, each $p(y_t \mid \mathbf{v}, y_1, \ldots, y_{t-1})$ distribution is represented with a softmax over all the words in the vocabulary.

<b> Objective Function </b>  
$$
\frac{1}{|\mathcal{S}|} \sum_{(T,S)\in \mathcal{S}} \log p(T \mid S)
$$
Note that objective function is same as that of RNN encoder-decoder (So it is same as that of RNN), where log probability of a correct translation $T$ given the source sentence $S$, while $\mathcal{S}$ is the trianing set.  
Once training is complete, produce translations by finding the most likely translation according to the LSTM,
$$
\hat{T} = \arg\max_T \, p(T \mid S)
$$

<b> Details of Seq2Seq Model </b>
1. Using Beam search: the translation using a simple left-to-right beam search decoder which
maintains a small number $B$ of partial hypotheses, where a partial hypothesis is a prefix of some translation.
2. Deep LSTMs: since found that deep LSTMs significantly outperformed shallow LSTMs, so we chose an LSTMwith four layers. 
3. Reverse order of input sequence: by reversing the words in the source sentence, the first few words in the source language are now very close to the first few words in the target language. So the backpropagation has an easier time “establishing communication” between
the source sentence and the target sentence, which in turn results in substantially improved overall performance. For example, mapping the sentence $a, b, c$ to the sentence $\alpha, \beta, \gamma$, the LSTM is asked to map  $c,b,a$ to  $\alpha, \beta, \gamma$
where $\alpha, \beta, \gamma$ is the translation of $a, b, c$. This way, $a$ is in close proximity to $\alpha$, $b$ is fairly close to $\beta$ and so on.

### Picking the most likely sentence
### Why not Greedy search?
## Beam Search
## Bleu Score

## Image Captioning
## Attention