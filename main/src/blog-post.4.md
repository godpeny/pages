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
Note that you're not trying to sample at random from the distribution $p$, instead, you want to find output sentence $y$ that maximizes the conditional probability $p$.
$$
\argmax_{y^{\langle 1 \rangle}, \ldots, y^{\langle T_y \rangle}} 
P\!\big(y^{\langle 1 \rangle}, \ldots, y^{\langle T_y \rangle} \mid x \big)
$$
Where $y^{\langle 1 \rangle}, \ldots, y^{\langle T_y \rangle} $ is output sentence sequence and $x$ is input sentence. For example, given the input French sentence, the model tells you what is the probability of different corresponding English translations.
```
Jane visite l’Afrique en septembre. (input sentence: French)

(output sentence: English)
→ Jane is visiting Africa in September.
→ Jane is going to be visiting Africa in September.
→ In September, Jane will visit Africa.
→ Her African friend welcomed Jane in September.
```
### Why not Greedy search?
Since you want to pick sequence of words that maximizes the joint probability. It turns out that picking up the best first word, best second word and so on; that approach doesn’t really work. It is not optimal to pick one best word at a time. 
 
Also the total number of combination of words in output sequence is exponentially large. For example, when you have $10,000$ vocabularies and try to make $10$ words long sentence, $P\!\big(y^{\langle 1 \rangle}, \ldots, y^{\langle 10 \rangle} \mid x \big)$ is $10,000^{10}$.  

Since the number is too large, the common approach is to use approximate search algorithm, which will try to pick the sentence that maximizes the conditional probability, such as Beam Search.

## Beam Search
Beam search is a heuristic search algorithm that explores a graph by expanding the most promising node in a limited set. Beam search is a modification of best-first search that reduces its memory requirements. It is a greedy algorithm since only a predetermined number($B$) of best partial solutions are kept as candidates. For example, in sequence to sequence model, decoder outputs the softmax output over all possibilities and keep the top $B$ picks in the memory in the initial step. In the second step, for each of $B$ choices, consider what should be the second word using the probability of first and second word.
$$
P(y^{\langle 1 \rangle}, y^{\langle 2 \rangle} \mid x) 
= P(y^{\langle 1 \rangle} \mid x) \; P(y^{\langle 2 \rangle} \mid x, y^{\langle 1 \rangle})
$$
Suppose $B=3$ and the number of vocabulary is $10,000$, then evaluate all $3 \times 10,000$ options according to the probability of first and second word, and pick top $B$.  
To generalize, at every step, instantiate $B$ copies of the network to evaluate the partial sentence fragments.

Unlike exact search algorithms like BFS (Breadth First Search) or DFS (Depth First Search), Beam Search runs faster but is not guaranteed to find exact maximum for $\arg\max_{y} p(y|x)$. If use larger $B$, better result, but slower. While use smaller $B$, worse result but faster.

### Refinement of Beam Search
<b> Original </b>
$$
\arg\max_{y} \prod_{t=1}^{T_y} P\!\big(y^{\langle t \rangle} \mid x, y^{\langle 1 \rangle}, \ldots, y^{\langle t-1 \rangle}\big)
$$
<b> Log Sum </b>  
Because multiplying tiny numbers result in numeric underflow, use sum of log.
$$
\arg\max_{y} \sum_{t=1}^{T_y} \log P\!\big(y^{\langle t \rangle} \mid x, y^{\langle 1 \rangle}, \ldots, y^{\langle t-1 \rangle}\big)
$$
<b> Log Sum + Normlization </b>  
Beam search might tend to prefer very short translations (outputs) because the probability for short sentence is determined by the fewer numbers which are less than $1$.
$$
\arg\max_{y} \frac{1}{T_y^{\alpha}} \sum_{t=1}^{T_y} \log P\!\big(y^{\langle t \rangle} \mid x, y^{\langle 1 \rangle}, \ldots, y^{\langle t-1 \rangle}\big)
$$
So normalize it with the number of the words then the product or sum of log will just be not as small. Note that $\alpha$ for softer normlization. When $\alpha=1$, complete normliazation, while $\alpha=0$ zero normlaization. By setting $0 < \alpha < 1$, setting normalization level somewhere between.
 
## Bleu Score
BLEU (bilingual evaluation understudy) is an algorithm for evaluating the quality of text which has been machine-translated from one natural language to another. Quality is considered to be the correspondence between a machine's output and that of a human.  
The central idea behind BLEU is that <b>"the closer a machine translation is to a professional human translation, the better it is"</b>.  
$$
\begin{array}{|c|c|c|c|c|c|c|c|}
\hline
\textbf{Candidate} & \text{the} & \text{the} & \text{the} & \text{the} & \text{the} & \text{the} & \text{the} \\
\hline
\textbf{Reference 1} & \text{the} & \text{cat} & \text{is} & \text{on} & \text{the} & \text{mat} & \\
\hline
\textbf{Reference 2} & \text{there} & \text{is} & \text{a} & \text{cat} & \text{on} & \text{the} & \text{mat} \\
\hline
\end{array}
$$
Let's see an above example to understand the algorithm of Bleu Score.
Of the seven words in the candidate translation, all of them appear in the reference translations. Thus the candidate text is given a unigram precision of,
${\displaystyle P={\frac {m}{w_{t}}}={\frac {7}{7}}=1}$, 
where $m$ is number of words from the candidate that are found in the reference, and $~w_{t}$ is the total number of words in the candidate. This is a perfect score, despite the fact that the candidate translation above retains little of the content of either of the references.

The modification that BLEU makes is fairly straightforward. For each word in the candidate translation, the algorithm takes its maximum total count in the single reference sentence, $~m_{max}$, in any of the reference translations. In the example above, the word "the" appears twice in reference 1, and once in reference 2. Thus $~m_{max}=2$.

For the candidate translation, the count $m_{w}$ of each word is clipped to a maximum of $m_{max}$ for that word. In this case, "the" has $~m_{w}=7$ and $~m_{max}=2$ thus $ ~m_{w}=2$. These clipped counts $~m_{w}$ are then summed over all distinct words in the candidate. This sum is then divided by the total number of unigrams in the candidate translation. In the above example, the modified unigram precision score would be $P={\frac {2}{7}}$.  
You can apply same logic to $n$-gram using the $n$ sets of words appearing next to each other.  
You can generalize as below.
$$
P_1 = \frac{\sum_{\text{unigrams} \in \hat{y}} \text{Count}_{\text{clip}}(\text{unigram})}
           {\sum_{\text{unigrams} \in \hat{y}} \text{Count}(\text{unigram})}\\[5pt]
P_n = \frac{\sum_{\text{n-grams} \in \hat{y}} \text{Count}_{\text{clip}}(\text{n-gram})}
           {\sum_{\text{n-grams} \in \hat{y}} \text{Count}(\text{n-gram})}
$$
Where $~m_{max}$ is count clip and $m$ is count is $w_t$.

The final version of Bleu score is below.
$$
BP \cdot \exp\!\left( \frac{1}{k} \sum_{n=1}^{k} P_n \right)
$$
Wher $P_n$ is  Bleu score on $n$-grams only and BP is,
$$
BP =
\begin{cases}
1 & \text{if } \text{model\_output\_length} > \text{reference\_output\_length} \\
\exp\!\left( 1 - \frac{\text{model\_output\_length}}{\text{reference\_output\_length}} \right) & \text{otherwise}
\end{cases}
$$
The Blue score unduly gives a high score for candidate strings that are containing all the $n$-grams of reference strings, but for as few times as possible. Simply speaking, short output translation is likely to get good precision. So brevity penalty is applied to punish candidate strings that are too short.

## Image Captioning
## Attention