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
At each time step, RNNs process units($h$) with a fixed activation function. These units have an internal hidden state that acts as memory that retains information from previous time steps. This memory allows the network to store past knowledge and adapt based on new inputs. So when making prediction for $y^{<k>}$ in RNN, it gets information from not only $x^{<k>}$ but also information from previous time steps $x^{<1>} \sim x^{<k-1>}$.

## Forward propagaion of RNN
![alt text](images/blog7_rnn_forward_propagation.png)
From the above picture describing the forward propagation of RNN, let's see how the forward propagation works in first layer of RNN. Note that instead of using parameters notation $W,V,U$, we use notation like $W_{aa}$ for parameters to make it more clear.
$$
a^{\langle 0 \rangle} = \vec{0} (\text{zero vector}) \\[5pt]
a^{\langle 1 \rangle} = g_1(W_{aa} a^{\langle 0 \rangle} + W_{ax} x^{\langle 1 \rangle} + b_a) \quad (g_1 = \text{tanh or ReLU}) \\[5pt]
\hat{y}^{\langle 1 \rangle} = g_2(W_{ya} a^{\langle 1 \rangle} + b_y)
\quad (g_2 = \text{sigmoid, activation depends on type of output}) 
$$
More generally, 
$$
a^{\langle t \rangle} = g(W_{aa} a^{\langle t-1 \rangle} + W_{ax} x^{\langle t \rangle} + b_a) \\[5pt]
\hat{y}^{\langle t \rangle} = g(W_{ya} a^{\langle t \rangle} + b_y)
$$
Note that a set of weights(parameters) uses for each time steps are shared. First, parameter $W_{ax}$ goverens the connection between $x$ and $a$ (hidden layer) on every time step, secondly, the horizontal activation connection is governend by $W_{aa}$ on every time step. Lastly, $W_{ya}$ goverens the connection between output and hidden layer on every time step.  
The notation convention behind is that for $W_{ax}$, the second index $x$ indicates that $W_{ax}$ will be multipied by some $x$ quantity and the first index $a$ indicates that $W_{ax}$ is used to compute output $a$ quantity. 
$$a \leftarrow W_{ax} x^{\langle 1 \rangle}$$

We can simplified the below RNN notation that first introduced above. 
![alt text](images/blog7_rnn_forward_propagation_simplified.png)
Suppose $a \in \mathbb{R}^{100}$ and $x \in \mathbb{R}^{10000}$, therefore $W_{aa} \in \mathbb{R}^{(100 \times 100)}$ and $W_{ax} \in \mathbb{R}^{(100 \times 10000)}$.
$$
a^{\langle t \rangle} = g(W_{aa} a^{\langle t-1 \rangle} + W_{ax} x^{\langle t \rangle} + b_a) \\
= g \left( W_a \begin{bmatrix} a^{\langle t-1 \rangle}, x^{\langle t \rangle} \end{bmatrix} + b_a \right)
$$
Let's understand the the simplified notation of $a^{\langle t \rangle}$ above. First, by putting two matrices $W_{aa}$ and $W_{ax}$ side by side horizontally. 
$$W_a = \left[ W_{aa} \, \big| \, W_{ax} \right] \in \mathbb{R}^{(100 \times 10100)}$$ 
Secondly, similarly, take two vector $a$ and $x$ and stack together.
$$
\left[ a^{\langle t-1 \rangle},\ x^{\langle t \rangle} \right]
= \left[\begin{array}{c} a^{\langle t-1 \rangle} \\ - \\ x^{\langle t \rangle}\end{array} \right] \in \mathbb{R}^{(10100)}
$$
To sum up, we can interprete as below.
$$
W_a \begin{bmatrix} a^{\langle t-1 \rangle}, x^{\langle t \rangle} \end{bmatrix} = \left[ W_{aa} \, \big| \, W_{ax} \right] \times \left[\begin{array}{c} a^{\langle t-1 \rangle} \\ - \\ x^{\langle t \rangle}\end{array} \right]
$$ 


## Back propagation of RNN (Backpropagation Through Time, BPTT)
![alt text](images/blog7_rnn_backward_propagation.png)

The loss function for each time step can be calculated as below, using cross-entropy loss.
$$
\mathcal{L}^{\langle t \rangle}(\hat{y}^{\langle t \rangle}, y^{\langle t \rangle}) = - y^{(t)} \log \hat{y}^{\langle t \rangle} - (1 - y^{(t)}) \log (1 - \hat{y}^{\langle t \rangle}) \\[5pt]
\mathcal{L}(\hat{y}, y) = \sum_{t=1}^{T_y} \mathcal{L}^{\langle t \rangle}(\hat{y}^{\langle t \rangle}, y^{\langle t \rangle})
$$
Just like backpropagation of standard neural network, take the derivatives with respect to all the parameters using gradient descent algorithm to the opposite direcions to all the forward propagation.  
One thing to note in the backpropagation is that the calculation from right to left, going backward in time steps. This special backprogation is called Backpropagation Through Time, BPTT.

## Types of RNNs
![alt text](images/blog7_types_of_rnn.png)
1. One-to-One RNN
This is the simplest type of neural network architecture where there is a single input and a single output. It is used for straightforward classification tasks such as binary classification where no sequential data is involved. (standard NN and no need to be RNN)

2. One-to-Many RNN
In a One-to-Many RNN the network processes a single input to produce multiple outputs over time. This is useful in tasks where one input triggers a sequence of predictions (outputs). For example music generation.

3. Many-to-One RNN
The Many-to-One RNN receives a sequence of inputs and generates a single output. This type is useful when the overall context of the input sequence is needed to make one prediction. In sentiment analysis the model receives a sequence of words (like a sentence) and produces a single output like positive, negative or neutral.

4. Many-to-Many RNN
The Many-to-Many RNN type processes a sequence of inputs and generates a sequence of outputs. In language translation task a sequence of words in one language is given as input and a corresponding sequence in another language is generated as output.


## Language Model
### Sampling Novel Sequences
## Vanishing Gradients with RNNs

## Gated Reccurent Unit (GRU)
## Long Short Term Memory Unit (LSTM)
## Bidirectional RNN
## Deep RNN