# Recurrent Neural Networks (RNN)
## Sequence Models
Sequence modeling trains neural network on chronologically ordered data to capture patterns and make predictions over time. More specifically, sequence models like recurrent neural networks process inputs as sequences, with each data point conditioned on those preceding it. The model iterates through data points, maintaining an encoded representation of the sequence history at each step. This sequential processing allows the model to learn complex time-based patterns like trends, seasonality, and long-range dependencies in data. The sequence model is trained to make predictions by estimating the probability distribution over next values, given the sequence of past context. This modeling of ordered data as interdependent steps enables the model to develop a sense of continuity and dynamics within data. By absorbing implicit rules about events unfolding over time, sequence models can gain limited foresight to make informed predictions about what may follow based on sequenced history. This time-based conditioning provides valuable context for inference compared to assessing data points independently. With extensive training over representative sequences, models can become skilled at leveraging the past to anticipate the future.

Sequence modeling is crucial to understand data that unfolds over time. Unlike static data, temporal sequences have complex time-based patterns like trends, cycles, and lagged effects. By processing data as interdependent sequenced steps, models can learn these nuanced time dynamics rather than viewing data points in isolation. This time-based conditioning enables models to make more contextual and accurate predictions and decisions — understanding how the past leads to the future. Sequence modeling has unlocked advancements in speech, text, video, forecasting, anomaly detection, and more.  

### Notation for Sequence Models - One Hot Encoding
The one-hot vector is a notation for describing training set(both $x,y$) for sequence data. It is a method for converting categorical variables into a binary format. It creates new columns for each category where $1$ means the category is present and $0$ means it is not. The primary purpose of One Hot Encoding is to ensure that categorical data can be effectively used in machine learning models. For example, in natural language processing, $(1 \times N)$ matrix (vector) used to distinguish each word in a vocabulary from every other word in the vocabulary. The vector consists of $0$ in all cells with the exception of a single $1$ in a cell used uniquely to identify the word. 

$$
\textbf{Label Encoding} \\
\begin{array}{|c|c|c|}
\hline
\textbf{Food Name} & \textbf{Categorical \#} & \textbf{Calories} \\
\hline
\text{Apple} & 1 & 95 \\
\text{Chicken} & 2 & 231 \\
\text{Broccoli} & 3 & 50 \\
\hline
\end{array} \\[5pt]

\textbf{One Hot Encoding} \\
\begin{array}{|c|c|c|c|}
\hline
\textbf{Apple} & \textbf{Chicken} & \textbf{Broccoli} & \textbf{Calories} \\
\hline
1 & 0 & 0 & 95 \\
0 & 1 & 0 & 231 \\
0 & 0 & 1 & 50 \\
\hline
\end{array}
$$
![alt text](images/blog7_one_hot_encodding.png)

## Basics of RNN
In artificial neural networks, recurrent neural networks (RNNs) are designed for processing sequential data, such as text, speech, and time series, where the order of elements is important. Since sequential data are stored in one hot vectors, we need to build a network that can learn the mapping from $x$ to $y$.  

There are several reasons why standard network are not used for sequential data.
 - Inputs, outputs can be different lengths in different examples.
- Doesn’t share features learned across different positions of text. For example, from the input data, if the word "Harry" appeared in earlier position, $x^{<1>}$, this gives a sign that "Harry" is a person's name. So it would be nice if the network figured out that "Harry" in other later position is also could be person's name.
- Also you can reduce computational cost by reducding the number of parameters in the model.

This is because, Unlike feedforward neural networks, which process inputs independently, RNNs utilize recurrent connections, where the output of a neuron at one time step is fed back as input to the network at the next time step. This enables RNNs to capture temporal dependencies and patterns within sequences.

### Architecture of RNN
![alt text](images/blog7_rnn_arch.png)
RNNs share similarities in input and output structures with other deep learning architectures but differ significantly in how information flows from input to output. Unlike traditional deep neural networks where each dense layer has distinct weight matrices. RNNs use shared weights across time steps, allowing them to remember information over sequences. So from the image above, weights $U,W,V$ are all same paramters shared across all time steps.  
At each time step, RNNs process units($h$) with a fixed activation function. These units have an internal hidden state that acts as memory that retains information from previous time steps. This memory allows the network to store past knowledge and adapt based on new inputs. So when making prediction for $y^{<k>}$ in RNN, it gets information from not only $x^{<k>}$ but also information from $x^{<1>} \sim x^{<k-1>}$.

### Forward propagaion of RNN
### Back propagation of RNN (Backpropagation Through Time, BPTT)

## Types of RNNs

## Language Model
### Sampling Novel Sequences
## Vanishing Gradients with RNNs

## Gated Reccurent Unit (GRU)
## Long Short Term Memory Unit (LSTM)
## Bidirectional RNN
## Deep RNN