# Natural Language Processing (NLP)
## Preliminaries
### Cosine Similarity
The cosine similarity is a measure of similarity between two non-zero vectors defined in an inner product space. Cosine similarity is the cosine of the angle between the vectors;that is, it is the dot product of the vectors divided by the product of their lengths.  
It follows that the cosine similarity does not depend on the magnitudes of the vectors, but only on their angle. The cosine similarity always belongs to the interval $[-1,+1]$.  
For example, two proportional vectors have a cosine similarity of +1, two orthogonal vectors have a similarity of 0, and two opposite vectors have a similarity of −1.
$$
\mathrm{sim}(u,v)=\frac{u^{\mathsf T}v}{\|u\|_2\,\|v\|_2}
$$
Cosine of the angle between two vectors $u$ and $v$.
Or you can use square distance(Euclidiean distance) to measure dissimilarity between two vectors.
$$
||u-v||^{2}
$$

## Word Representation
### One-Hot vector (encoding)
A one-hot vector is a $1 \times N$ matrix (vector) used to distinguish each word in a vocabulary from every other word in the vocabulary. The one-hot vector consists of 0s in all cells with the exception of a single 1 in a cell used uniquely to identify the word.
$$
\begin{array}{cccccc}
\text{Man (5391)} & \text{Woman (9853)} & \text{King (4914)} &
\text{Queen (7157)} & \text{Apple (456)} & \text{Orange (6257)} \\[6pt]
\left[\!\!\begin{array}{c}
0\\[2pt]0\\[2pt]0\\[2pt]\vdots\\[2pt]1\\[2pt]\vdots\\[2pt]0\\[2pt]0\\[2pt]0
\end{array}\!\!\right] &
\left[\!\!\begin{array}{c}
0\\[2pt]0\\[2pt]0\\[2pt]\vdots\\[2pt]1\\[2pt]\vdots\\[2pt]0\\[2pt]0\\[2pt]0
\end{array}\!\!\right] &
\left[\!\!\begin{array}{c}
0\\[2pt]0\\[2pt]\vdots\\[2pt]1\\[2pt]\vdots\\[2pt]0\\[2pt]0\\[2pt]0\\[2pt]0
\end{array}\!\!\right] &
\left[\!\!\begin{array}{c}
0\\[2pt]0\\[2pt]0\\[2pt]0\\[2pt]\vdots\\[2pt]1\\[2pt]\vdots\\[2pt]0\\[2pt]0
\end{array}\!\!\right] &
\left[\!\!\begin{array}{c}
0\\[2pt]\vdots\\[2pt]1\\[2pt]0\\[2pt]0\\[2pt]0\\[2pt]\vdots\\[2pt]0\\[2pt]0
\end{array}\!\!\right] &
\left[\!\!\begin{array}{c}
0\\[2pt]0\\[2pt]0\\[2pt]0\\[2pt]\vdots\\[2pt]0\\[2pt]\vdots\\[2pt]1\\[2pt]0
\end{array}\!\!\right] \\[6pt]
\mathbf{o}_{5391} & \mathbf{o}_{9853} & \mathbf{o}_{4914} &
\mathbf{o}_{7157} & \mathbf{o}_{456} & \mathbf{o}_{6257}
\end{array}
$$

### Word Embeddings
Word embeddings are a way of representing words as vectors in a multi-dimensional space, where the distance and direction between vectors reflect the similarity and relationships among the corresponding words.   
Word embeddings are trained using machine learning techniques, often based on neural networks. The idea is to learn representations that encode semantic meaning and relationships between words. Word embeddings are trained by exposing a model to a large amount of text data and adjusting the vector representations based on the context in which words appear.
$$
\begin{array}{c|rrrrrr}
\textbf{Feature}
& \textbf{Man\ (5391)}
& \textbf{Woman\ (9853)}
& \textbf{King\ (4914)}
& \textbf{Queen\ (7157)}
& \textbf{Apple\ (456)}
& \textbf{Orange\ (6257)}
\\ \hline
\text{Gender} & -1.00 &  1.00 & -0.95 &  0.97 &  0.00 &  0.01 \\
\text{Royal}  &  0.01 &  0.62 & \underline{0.93} & \underline{0.95} & -0.01 &  0.00 \\
\text{Age}    &  0.03 &  0.02 &  0.70 &  0.69 &  0.03 & -0.02 \\
\text{Food}   &  0.04 &  0.01 &  0.02 &  0.01 &  0.95 &  0.97 \\
\vdots        & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots
\end{array}
$$
For example, a lot of features of 'apple' and 'orange' are similar.

## Named Entity Recognition(NER)
Named Entity Recognition is that identifying and categorizing important information known as entities in text. These entities can be names of people, places, organizations, dates, etc.  
For example, see below example. An annotated block of text that highlights the names of entities.
$$
\text{Jim bought 300 shares of Acme Corp.\ in 2006.} \\[5pt]
\text{[Jim]}_{\text{Person}}
\ \text{bought 300 shares of }
\text{[Acme Corp.]}_{\text{Organization}}
\ \text{in }
\text{[2006]}_{\text{Time}}
\text{.}
$$
## Transfer Learning with Word Embedding
1. Learn word embeddings from large text corpus. (1-100B words)
(Or download pre-trained embedding online.)
2. Transfer embedding to new task with smaller training set.
(say, 100k words)
3. Optional: Continue to finetune the word embeddings with new
data. (When above task 2 has a pretty big data set)

## Analogy Reasoning
A surprising property of word vectors is that word analogies can often be solved with vector arithmetic. At the core of analogy reasoning is the idea that differences between word vectors capture meaningful relationships.  
For example, consider below word embeddings. 
$$
\begin{array}{c|rrrrrr}
 & \textbf{Man\ (5391)} & \textbf{Woman\ (9853)} &
   \textbf{King\ (4914)} & \textbf{Queen\ (7157)} &
   \textbf{Apple\ (456)} & \textbf{Orange\ (6257)}\\ \hline
\text{Gender} & -1.00 &  1.00 & -0.95 & 0.97 & 0.00 & 0.01 \\
\text{Royal}  &  0.01 &  0.02 &  0.93 & 0.95 & -0.01 & 0.00 \\
\text{Age}    &  0.03 &  0.02 &  0.70 & 0.69 & 0.03 & -0.02 \\
\text{Food}   &  0.09 &  0.01 &  0.02 & 0.01 & 0.95 & 0.97
\end{array}
$$
Let's say we want to find out what '?' is from below classic analogy using above embedding.
$$
\text{man : woman :: king : ?}
$$
The key observation is that the vector difference. Both $\mathbf{e}_{\text{man}} - \mathbf{e}_{\text{woman}}$ and $\mathbf{e}_{\text{king}} - \mathbf{e}_{\text{queen}}$ result in vectors with similar properties. By comparing these differences, we can automatically infer that replacing the gender dimension in “king” should yield “queen”.
$$
\mathbf{e}_{\text{man}} - \mathbf{e}_{\text{woman}}
\;\approx\;
\mathbf{e}_{\text{king}} - \mathbf{e}_{\text{queen}}
\;\approx\;
\begin{bmatrix}
-2\\[2pt] 0\\[2pt] 0\\[2pt] 0\\[2pt] \vdots
\end{bmatrix}
$$

Let's formalize and turn it into an algorithm.
$$
\mathbf e_{\text{man}}-\mathbf e_{\text{woman}} \;\approx\; \mathbf e_{\text{king}}-\mathbf e_{\text{?}}
$$
To find the word embedding $e_{\text{?}}$, use cosine similarity.
$$
\arg\max_{w}\;
\operatorname{sim}\!\big(
\mathbf e_{?},\;
\mathbf e_{\text{king}}-\mathbf e_{\text{man}}+\mathbf e_{\text{woman}}
\big)
$$
One of the remarkable results about word embeddings is the generality of analogy relationship they can learn. And all the analogies can be learned just by running a word embedding learning algorithm on the large text corpus.

## Embedding Matrix
When you implement an algorithm to learn word embedding, what you end up learning is an embedding matrix.
An embedding matrix refers to a matrix used to map each word in the training data to a word-embedding vector, allowing for a lower-dimensional representation of the vocabulary. It is collaboratively optimized during the training process to improve the model's performance.
![alt text](images/blog3_word_embedding_matrix.png)  
$$
\text{(word embedding matrix)} \times \text{(one hot vector)} = \text{(selecting out the column corresponding to the word)}
$$

## Neural Language Model (N-gram Language Model)
N-gram models predict the probability of a word given the previous n−1 words.
![alt text](images/blog3_ngram_language_model.png)  
The goal is to predict the word in the sentence, "I want a glass of orange __".  
1. Construct one-hot vector $o$ for the words and embedding matrix $E$.
2. Multiply embedding matrix with each word's one-hot vector to select out the embedding vector for the words.
3. Feed the embedding vectors to neural network.
4. The output of the neural network feeds to softmax network.
5. The softmax classifies among the vocabularies. 

What commonly done is to have a fixed historial window which means that you predict the next word given previous $n$ words.  
For example, if you choose $n=4$, you only need word vectors of 'a', 'glass', 'of', 'orange'.

### Other Context/Target Pairs
- $n$ words on left and right side of the target word.
- Last $1$ word
- Nearby $1$ word

The researchers found that it’s natural to use last few words as context to build a language model. But if your main goal is to learn word embedding, then you have to use all the other contexts to build a meaningful word embedding as well. (Word2Vec Model)

## Word2Vec
### Skip-gram
#### Gradient of Skip-gram

## Negative Sampling
### Gradient of Negative Sampling
### C-Bow
### GloVE

## Sentiment Classification
## Debiasing Word Embeddings

