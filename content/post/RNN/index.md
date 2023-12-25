+++
author = "Laychiva Chhout"
title = "Recurrent Neural Networks"
date = "2023-12-24"
description = "Recurrent Neural Networks in detail."
math = "true"
tags = [
    "ai",
    "ml",
    "dl",
]
categories = [
    "Artificial Intelligence",
    "Deep Learning",
]
series = ["Themes Guide"]
aliases = ["migrate-from-jekyl"]
image = "pawel-czerwinski-8uZPynIu-rQ-unsplash.jpg"
+++

{{< math.inline >}}
{{ if or .Page.Params.math .Site.Params.math }}
<!-- KaTeX -->
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.11.1/dist/katex.min.css" integrity="sha384-zB1R0rpPzHqg7Kpt0Aljp8JPLqbXI3bhnPWROx27a9N0Ll6ZP/+DiW/UqRcLbRjq" crossorigin="anonymous">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.11.1/dist/katex.min.js" integrity="sha384-y23I5Q6l+B6vatafAwxRu/0oK/79VlbSz7Q9aiSZUvyWYIYsd+qj+o24G5ZU2zJz" crossorigin="anonymous"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.11.1/dist/contrib/auto-render.min.js" integrity="sha384-kWPLUVMOks5AQFrykwIup5lo0m3iMkkHrD0uJ4H5cjeGihAutqP0yW0J6dpFiVkI" crossorigin="anonymous" onload="renderMathInElement(document.body);"></script>
{{ end }}
{{</ math.inline >}}


# Recurrent Neural Network

Disclaimer:
This note is based on a RNN lecture by Geoffroy Peeters, Télécom Paris, IP Paris. 

 While I have done my best to ensure that the information in this note is accurate and up-to-date, I cannot guarantee that it is perfect. Please refer to the original sources for more detailed information and for any updates that may have been made since the writing of this note.
 
 Additionally, I would like to emphasize that this note is for educational purposes only and should not be taken as professional advice.
# Introduction

When thinking of solving a problem, our brain always search for the information that we have experienced in the past and not solve a problem from scratch or without any information. You are reading this text and you are able to understand this text because of the previous words. 

The Feed Forward Neural Network or Deep Feed Forward Neural Network can not do this because the result of the network is given by only a hidden state and apply the activation function on it. For example, if you want to predict a stock price, you can't only use the data from yesterday to predict the stock price of today, you might need the older data as well. 

Recurrent Neural Network is here to deal with this issue, it can deal with the sequential data and remember the all the previous steps to produce the result. In order to do so, RNN has it inner memory or we call it loops or folds. 
# Notation

Inputs:

- $\mathbf{x}^{\langle t\rangle}$ : input vector $\mathbf{x}$ at time $t$ (of dimension $n^{[0]}$) 
- $\mathbf{x}^{\langle 1\rangle}, \mathbf{x}^{\langle 2\rangle}, \ldots, \mathbf{x}^{\left\langle T_x\right\rangle}$ : a sequence of inputs of length $T_x$ 

Outputs:
- $\mathbf{y}^{\langle t\rangle}$ : output vector $\mathbf{y}$ at time $t$ 

- $\mathbf{y}^{\langle 1\rangle}, \mathbf{y}^{\langle 2\rangle}, \ldots, \mathbf{y}^{\left\langle T_y\right\rangle}$ : a sequence of outputs of length $T_y$ 

One thing to note here is that the length of the input and output sequence could be different.
## Type of RNN

### Many to Many $T_x = T_y$

If $T_x = T_y$ the input sequence and output sequence have the same length. For example, the problem of Name Entity Relation, which we try to define type of words in a sentence. 

| Input $\underline{x}$ | Output $\underline{y}$ | Type | Examples |
| :--- | :--- | :--- | :--- |
| sequence $T_x1$ | sequence $T_y=T_x$ | Many-To-Many | Named entity recognition |

![Alt text](/Photos/rnn1.png)
<!-- ![[./Photos/Screenshot 2023-10-04 at 20.26.28.png | This is the caption]] -->

### Many to One: $T_x=1, T_y =1$


| Input $\underline{x}$ | Output $\underline{y}$ | Type | Examples |
| :--- | :--- | :--- | :--- |
| sequence $T_x = 1$ | single $T_y=1$ | Many-To-One | Sentiment analysis, Video activity detection |

![Alt text](/Photos/rnn2.png)
<!-- ![[../Photos/Screenshot 2023-10-04 at 20.31.02.png]] -->

**Example**: Sentiment Analysis that takes a document as an input and give the sentiment as the result.

### One to Many: $T_x = 1, T_y 1$


| Input $\underline{x}$ | Output $\underline{y}$ | Type | Examples |
| :--- | :--- | :--- | :--- |
| single $T_x=1$ | sequence, $T_y1$ | One-To-Many | Text generation, music generation |

![Alt text](/Photos/rnn2_2.png)

<!-- ![[../Photos/Screenshot 2023-
10-04 at 20.33.18.png]] -->

### Many to Many $T_x \neq T_y$ 

| Input $\underline{\underline{x}}$ | Output $\underline{\underline{y}}$ | Type         | Examples                                                                                               |
|:--------------------------------- |:---------------------------------- |:------------ |:------------------------------------------------------------------------------------------------------ |
| sequence $T_x1$                  | sequence $T_y1, T_y \neq T_x$     | Many-To-Many | $\begin{array}{l}\text { Automatic Speech Recognition, Ma- } \\\text { chine translation }\end{array}$ |

![Alt text](/Photos/rnn3.png)
<!-- ![[../Photos/Screenshot 2023-10-04 at 20.35.24.png]] -->

**Example:** DeepL Translator.

# But what is the RNN?

 **Question** 
 Why not using a Multi-Layer-Perceptron?

 **Answer**
- We could indeed represent an input sequence $\mathbf{x}^{\langle 1\rangle}, \ldots, \mathbf{x}^{\left\langle T_x\right\rangle}$ as a large-vector of dimension $\left(n^{[0]} T_x\right)$
	- it would requires a huge input dimension!

 **Problem**
 - $T_x^{(i)}$ can be different for each $i$, 
 - $\Rightarrow$ the dimension $\left(n^{[0]} \boldsymbol{T}_x^{(i)}\right)$ would be different for each $i$
 	
 **Problem**
- zero-padding ? $\left(n^{[0]} \max _i T_x^{(i)}\right)$
- it would still requires a huge input dimension!

 **Another problem**
 - the network will have to learn different weights for the same dimension $n^{[0]}$ at various time $\langle t\rangle$ in the sequence $\Rightarrow$ no weight sharing !
 - For example: 
 	- Emmanuel .... - learn for 1st position
 	- Currently Emmanuel .... - learn for 2nd position

**Solution** 
- Recurrent Neural Network
- 1D ConvNet (convolution only over time)

Basically, a RNN is a neural network with a roll inside or loop inside. 

![Alt text](/Photos/rnn4.png)
<!-- ![[../Photos/Screenshot 2023-10-04 at 20.38.28.png]] -->
Source: http://colah.github.io/posts/2015-08-Understanding-LSTMs/

The RNN is simply the neural network but for each step $t0$, it takes the information from the previous step to generate a result at the current step. 

![[../Photos/Screenshot 2023-10-04 at 20.40.46.png]]
There are three different weight inside this RNN, 
- $\mathbf{W}_{a x}$ from $\mathbf{x}^{\langle t\rangle} \rightarrow \mathbf{a}^{\langle t\rangle}$,
- $\mathbf{W}_{a a}$ from $\mathbf{a}^{\langle t-1\rangle} \rightarrow \mathbf{a}^{\langle t\rangle}$,
- $\mathbf{W}_{y a}$ from $\mathbf{a}^{\langle t\rangle} \rightarrow \mathbf{y}^{\langle t\rangle}$
We can see those weights as below:

![Alt text](/Photos/rnn5.png)
<!-- ![[../Photos/Screenshot 2023-10-04 at 20.41.36.png]] -->

We can see that $W_{aa}$ is actually the same across all the steps, it is a shared weight inside the loop. 
Where $a^{<t} = g_1(W_{aa}a^{<t-1}+W_{ax}x^{<t}+b_a)$ which means the information that is sent to the current step come from the information that the previous step received from the previous-previous step and the input at the previous step.

  Result of each timestep
The result of each step is obtained by: $\hat{\mathbf{y}}^{\langle t\rangle}=g_2\left(\mathbf{W}_{y a} \mathbf{a}^{\langle t\rangle}+\mathbf{b}_y\right)$

# Architecture of the RNN

## Standard RNN

![Alt text](/Photos/rnn7.png)
<!-- ![[../Photos/Screenshot 2023-10-04 at 20.51.21.png]] -->
A standard RNN will read the input from the left to right. 
But what will the RNN do if we want to predict the first word of the sentence, the RNN has no context of the full sentence how can it generates the correct result? - **Bi-directional RNN**.

## Bi-directional RNN

![Alt text](/Photos/rnn6.png)
<!-- ![[../Photos/Screenshot 2023-10-04 at 20.50.47.png]] -->

This RNN simply reads the input from the left to right and from the right to left and the prediction is done from the concatenation of $\overrightarrow{a}^{\langle t\rangle}$ and $\overleftarrow{a}^{\langle t\rangle}$ which is given by the following formula:

$$\hat{y}^{\langle t\rangle}=g\left(W_y\left[\vec{a}^{\langle t\rangle}, \overleftarrow{a}^{\langle t\rangle}\right]+b_y\right)$$
## Bidirectional RNN - Stanford Slides

Bi-directional RNN can be explained as below:

![Alt text](/Photos/rnn8.png)
<!-- ![[../Photos/Screenshot 2023-10-12 at 11.54.46.png | center | 500]] -->

![Alt text](/Photos/rnn9.png)
<!-- ![[../Photos/Screenshot 2023-10-12 at 11.55.34.png |center | 500]] -->

Limitation of Bidirectional RNN
- Note: bidirectional RNNs are only applicable if you have access to the entire input sequence
- They are not applicable to Language Modeling, because in LM you only have left context available.
- If you do have entire input sequence (e.g., any kind of encoding), bidirectionality is powerful (you should use it by default).
- For example, BERT (Bidirectional Encoder Representations from Transformers) is a powerful pretrained contextual representation system built on bidirectionality.

## Deep-RNN or Multi-Layer RNN or Stacked RNN

![Alt text](/Photos/rnn10.png)

<!-- ![[../Photos/Screenshot 2023-10-04 at 20.54.49.png|center | 500]] -->

- For the layer [1], the inputs of the cell at time $\langle t\rangle$ are
	- the input at the current time $\langle t\rangle: x^{\langle t\rangle}=\mathbf{a}^{[0]\langle t\rangle}$
	- the value of the cell at the previous time $\langle t-1\rangle: \mathbf{a}^{[1]\langle t-1\rangle}$
- For the layer $[l]$, the inputs of the cell at time $\langle t\rangle$ are
	- the value of the cell of the previous layer $[l-1]$ at the current time $\langle t\rangle: \mathbf{a}^{[l-1]\langle t\rangle}$
	- the value of the cell of the current layer $[l]$ at the previous time $\langle t-1\rangle: \mathbf{a}^{[l]\langle t-1\rangle}$
	$$a^{[l]<t}=g\left(W_a^{[l]}\left[a^{[l]\langle t-1\rangle}, a^{[l-1]\langle t\rangle}\right]+b_a^{[l]}\right)$$
### Multi-Layer RNN in Practice 

- Multi-layer or stacked RNNs allow a network to compute more complex representations 
	- they work better than just have one layer of high-dimensional encodings!
	- The lower RNNs should compute lower-level features and the higher RNNs should compute higher-level features.
- High-performing RNNs are usually multi-layer (but aren't as deep as convolutional or feed-forward networks)
- For example: In a 2017 paper, Britz et al.[Massive Exploration of Neural Machine Translation Architecutres”, Britz et al, 2017. https://arxiv.org/pdf/1703.03906.pdf] find that for Neural Machine Translation, 2 to 4 layers is best for the encoder RNN, and 4 layers is best for the decoder RNN
	- Often 2 layers is a lot better than 1 , and 3 might be a little better than 2
	- Usually, skip-connections/dense-connections are needed to train deeper RNNs (e.g., 8 layers)
- Transformer-based networks (e.g., BERT) are usually deeper, like 12 or 24 layers.
# Back Propagation Through Time (BPTT)

![Alt text](/Photos/rnn10.png)
<!-- ![[../Photos/Screenshot 2023-10-04 at 20.56.47.png  |center | 500]] -->

We can clearly see the problem here during the back-propagation through time of a RNN which is the **vanishing** and **exploding** gradients. 
Observing the formula:
$$
\begin{aligned}
\frac{\partial \mathscr{L}^{\langle t\rangle}}{\partial W_{a a}}=
& \sum_{k=0}^t\left(\prod_{i=k+1}^t \frac{\partial a^{\langle i\rangle}}{\partial a^{\langle i-1\rangle}}\right) \frac{\partial a^{\langle k\rangle}}{\partial W_{a a}}
\end{aligned}
$$
- Suppose only one hidden units, then $a^{\langle t\rangle}$ is a scalar and consequently $\frac{\partial a^{\langle t\rangle}}{\partial a^{\langle t-1\rangle}}$ is also a scalar
	- if $\left|\frac{\partial a^{\langle t\rangle}}{\partial a^{\langle t-1\rangle}}\right|<1$, then the product goes to $\mathbf{0}$ exponentially fast
	- if $\left|\frac{\partial a^{\langle t\rangle}}{\partial a^{\langle t-1\rangle}}\right| > 11$, then the product goes to $\infty$ exponentially fast
- Vanishing gradients: $\left|\frac{\partial a^{\langle t\rangle}}{\partial a^{\langle t-1\rangle}}\right|<1$
	- contributions from faraway steps vanish and don't affect the training
	- difficult to learn long-range dependencies
- Exploding gradients: $\left|\frac{\partial a^{\langle t\rangle}}{\partial a^{\langle t-1\rangle}}\right| > 1$
	- make the learning process unstable
	- gradient could even become $\mathbf{N a N}$

**How to deal with exploding gradient?
1. **Gradient clipping**
	gradient $d \theta=\frac{\partial \mathscr{L}}{\partial \theta}$ where $\theta$ are all the parameters
	- if $\|\theta\|$ > threshold
$$
d \theta \leftarrow \frac{\text { threshold }}{\|\theta\|} d \theta
$$
	- clipping doesn't change the direction of the gradient but change its length
	- we can clip only the norm of the part which causes the problem
	- we choose the threshold manually: start with a large threshold and then reduce it.
**Pseudo code**

![Alt text](/Photos/rnn12.png)
<!-- ![[../Photos/Screenshot 2023-10-12 at 12.07.44.png|center | 300]] -->

2. **Truncated BPTT**
- Training very long sequences
	- time consuming!
	- exploding gradients!
- Truncated BPTT:
	- run forward and backward passes through the chunks of the sequence (instead of the whole sequence)
- Forward
	- We carry hidden states forward in time forever
- Backward
	- only back-propagate in the chunks (small number of steps)
- Much faster but _dependencies longer than the chunk size don't affect the training_ (at least they still work at Forward pass) (Training is useless in that case). 

**How to deal with vanishing gradient?**
In RNN, at each step, we multiply the Jacobian matrix which leads to vanishing gradient if we have a long dependencies.
**Solution**: add a short-cuts between hidden states that are separated by more than one step.
![Alt text](/Photos/rnn13.png)
<!-- ![[../Photos/Screenshot 2023-10-04 at 21.10.57.png|center | 500]] -->
- Back-propagation through the shortcuts: the gradients vanish slower
	- we can learn long-term dependencies

### Vanishing gradient intuition (Stanford Slides):

![Alt text](/Photos/rnn14.png)
<!-- ![[../Photos/Screenshot 2023-10-12 at 11.38.18.png |center| 500]] -->


**Vanishing gradient proof sketch (linear case)**
- Recall:


$$
\boldsymbol{h}^{(t)}=\sigma\left(\boldsymbol{W}_h \boldsymbol{h}^{(t-1)}+\boldsymbol{W}_x \boldsymbol{x}^{(t)}+\boldsymbol{b}_1\right)
$$

- What if $\sigma$ were the identity function, $\sigma(x)=x$ ?

$$
\begin{aligned}
\frac{\partial \boldsymbol{h}^{(t)}}{\partial \boldsymbol{h}^{(t-1)}} & =\operatorname{diag}\left(\sigma^{\prime}\left(\boldsymbol{W}_h \boldsymbol{h}^{(t-1)}+\boldsymbol{W}_x \boldsymbol{x}^{(t)}+\boldsymbol{b}_1\right)\right) \boldsymbol{W}_h \\
& =\boldsymbol{I} \boldsymbol{W}_h=\boldsymbol{W}_h
\end{aligned}
$$


- Consider the gradient of the loss $J^{(i)}(\theta)$ on step $i$, with respect to the hidden state $\boldsymbol{h}^{(j)}$ on some previous step $j$. Let $\ell=i-j$

$$
\begin{aligned}
& \frac{\partial J^{(i)}(\theta)}{\partial \boldsymbol{h}^{(j)}}= \frac{\partial J^{(i)}(\theta)}{\partial \boldsymbol{h}^{(i)}} \prod_{j<t \leq i} \frac{\partial \boldsymbol{h}^{(t)}}{\partial \boldsymbol{h}^{(t-1)}} \\
&=\frac{\partial J^{(i)}(\theta)}{\partial \boldsymbol{h}^{(i)}} \prod_{j<t \leq i} W_h=\frac{\partial J^{(i)}(\theta)}{\partial \boldsymbol{h}^{(i)}} \prod_h \quad \text { (chain rule) }
\end{aligned}
$$


If $W_h$ is "small", then this term gets exponentially problematic as $\ell$ becomes large.

![alt text](/Photos/Screenshot%202023-10-12%20at%2011.46.57.png)
<!-- ![[../Photos/Screenshot 2023-10-12 at 11.46.57.png |center| 500]] -->
# LSTM and GRU


Long Short Term Memory networks – usually just called “LSTMs” – are a special kind of RNN, capable of learning long-term dependencies. They were introduced by [Hochreiter & Schmidhuber (1997)](http://www.bioinf.jku.at/publications/older/2604.pdf), and were refined and popularized by many people in following work.[1](http://colah.github.io/posts/2015-08-Understanding-LSTMs/#fn1) They work tremendously well on a large variety of problems, and are now widely used.
LSTMs are explicitly designed to avoid the long-term dependency problem. Remembering information for long periods of time is practically their default behavior, not something they struggle to learn!
All recurrent neural networks have the form of a chain of repeating modules of neural network. In standard RNNs, this repeating module will have a very simple structure, such as a single tanh layer.
![alt text](/Photos/LSTM3-SimpleRNN%201.png)
<!-- ![[../Photos/LSTM3-SimpleRNN 1.png |center| 500]] -->
LSTMs also have this chain like structure, but the repeating module has a different structure. Instead of having a single neural network layer, there are four, interacting in a very special way.
![alt text](/Photos/LSTM3-chain.png)
<!-- ![[../Photos/LSTM3-chain.png |center| 500]] -->

## The Core Idea Behind LSTMs

The key to LSTMs is the cell state, the horizontal line running through the top of the diagram.

The cell state is kind of like a conveyor belt. It runs straight down the entire chain, with only some minor linear interactions. It’s very easy for information to just flow along it unchanged.

![alt text](/Photos/LSTM3-C-line%201.png)
<!-- ![[../Photos/LSTM3-C-line 1.png]] -->
The LSTM does have the ability to remove or add information to the cell state, carefully regulated by structures called gates.

Gates are a way to optionally let information through. They are composed out of a sigmoid neural net layer and a pointwise multiplication operation.
The sigmoid layer outputs numbers between zero and one, describing how much of each component should be let through. A value of zero means “let nothing through,” while a value of one means “let everything through!”

An LSTM has three of these gates, to protect and control the cell state.
## Step-by-Step LSTM Walk Through

The first step in our LSTM is to decide what information we’re going to throw away from the cell state. This decision is made by a sigmoid layer called the “forget gate layer.” It looks at $h_{t−1}$ and $x_t$, and outputs a number between 0 and 1 for each number in the cell state $C_{t−1}$. A 1 represents “completely keep this” while a 0 represents “completely get rid of this.”

- **Action**:
	- The sigmoid (forget gate) answers for each component :
		- 1 : to keep it,
		- 0 to forget it, or
		- a value in-between to mitigate its influence
**- Intuition for language modelling**
	- The cell state might embed the gender of the present subject :
	- keep it to predict the correct pronouns,
	- or forget it, when a new subject appears
![alt text](/Photos/LSTM3-focus-f%201.png)
<!-- ![[../Photos/LSTM3-focus-f 1.png]] -->

The next step is to decide what new information we're going to store in the cell state. This has two parts. First, a sigmoid layer called the "input gate layer" decides which values we'll update. Next, a tanh layer creates a vector of new candidate values, $\tilde{C_t}$, that could be added to the state. In the next step, we'll combine these two to create an update to the state.

- **Action**
	- Create the update $\tilde{C_t}$ of the cell state
	- and its contribution $i_t$ (the update gate with a sigmoid activation)
- **Intuition for language modelling**
	- Add the gender of the new subject to the cell state,
	- to replace the old one we're forgetting.
![alt text](/Photos/LSTM3-focus-i%201.png)
<!-- ![[../Photos/LSTM3-focus-i 1.png]] -->

It's now time to update the old cell state, $C_{t-1}$, into the new cell state $C_t$. The previous steps already decided what to do, we just need to actually do it.

We multiply the old state by $f_t$, forgetting the things we decided to forget earlier. Then we add $i_t * \tilde{C_t}$. This is the new candidate values, scaled by how much we decided to update each state value.

- **Action**
	- Merge the old cell state modified by the forget gate with the new input
- **Intuition for language modelling**
	- Decide to drop the information about the old subject
	- Refresh the memory

![alt text](/Photos/LSTM3-focus-C%201.png)
<!-- ![[../Photos/LSTM3-focus-C 1.png]] -->

Finally, we need to decide what we're going to output. This output will be based on our cell state, but will be a filtered version. First, we run a sigmoid layer which decides what parts of the cell state we're going to output. Then, we put the cell state through tanh (to push the values to be between -1 and 1 ) and multiply it by the output of the sigmoid gate, so that we only output the parts we decided to.

- **Action**
	- Decide what parts of the (filtered) cell state to output $o_t$
	- Compute the hidden state
- **Intuition for language modelling**
	- Since we just saw a subject,
	- output the relevant information for the future (gender, number)
![alt text](/Photos/LSTM3-focus-o%201.png)
<!-- ![[../Photos/LSTM3-focus-o 1.png]] -->

## Recap LSTM:
![alt text](/Photos/Screenshot%202023-10-12%20at%2011.49.03.png)
<!-- ![[../Photos/Screenshot 2023-10-12 at 11.49.03.png |center| 500]] -->
![alt text](/Photos/Screenshot%202023-10-12%20at%2011.50.37.png)
<!-- ![[../Photos/Screenshot 2023-10-12 at 11.50.37.png]] -->

## GRU (Gated Recurrent Unit)

A slightly more dramatic variation on the LSTM is the Gated Recurrent Unit, or GRU, introduced by [Cho, et al. (2014)](http://arxiv.org/pdf/1406.1078v3.pdf). It combines the forget and input gates into a single “update gate.” It also merges the cell state and hidden state, and makes some other changes. The resulting model is simpler than standard LSTM models, and has been growing increasingly popular.

![alt text](/Photos/LSTM3-var-GRU.png)
<!-- ![[../Photos/LSTM3-var-GRU.png]] -->


# Application

## Language Model

Application: 
- Automatic Speech Recognition, Machine Translation, Optical Character Recognition
Language model: $p(sentence)$
- $p(y^{\langle 1\rangle},y^{\langle 2\rangle},\cdots,y^{\langle T_y\rangle})$
###### Notation
- $\mathbf{w}:=w_1^L:=w_1, w_2, \ldots w_L$
###### Goal:
- Estimate the (non-zero) probability of a word sequence for a given vocabulary
$$
\begin{aligned}
P\left(w_1^L\right) & =P\left(w_1, w_2, \ldots, w_L\right) \\
& =\prod_{i=1}^L P\left(w_i \mid w_1^{i-1}\right) \quad \forall i, w_i \in V \\
& =P\left(w_1\right) P\left(w_2 \mid w_1\right) P\left(w_3 \mid w_1, w_2\right) P\left(w_4 \mid w_1, w_2, w_3\right) \ldots P\left(w_L \mid w_1 \ldots w_{L-1}\right)
\end{aligned}
$$
- Simplification: the $\mathbf{N}$-gram assumption : use ($n-1$) previous words to predict the current word.
$$
P\left(w_1^L\right)=\prod_{i=1}^L P\left(w_i \mid w_{i-N+1}^{i-1}\right), \quad \forall i, w_i \in V
$$
- Bi-gram ( $\mathrm{N}=2)$ assumption
$$
P\left(w_1^L\right)=P\left(w_1\right) P\left(w_2 \mid w_1\right) P\left(w_3 \mid w_2\right) P\left(w_4 \mid w_3\right) \ldots P\left(w_L \mid w_{L-1}\right)
$$
- using Recurrent Neural Network (RNN)
$$
P\left(w_i \mid w_1^{i-1}\right)
$$
###### Training using a RNN

- **Algorithm**
	- Start with empty context, empty input word
	- Given the previous word as input $x^{\langle t \rangle}$ and the context (given by the hidden state cell $a^{\langle t\rangle}$) predict (maximise the probability to observe) the next word $y^{\langle t\rangle}$ as output $\hat{y}^{\langle t\rangle}$
- **Example**
![alt text](/Photos/Screenshot%202023-10-05%20at%2009.35.14.png)

<!-- ![[../Photos/Screenshot 2023-10-05 at 09.35.14.png]] -->
 Sentence = "French people work an average of 35 hours a week".

- **Training**
	- $\hat{y}^{\langle t\rangle}$ : output with a softmax with $K$ neurons, gives the probability of each of the $K$ words in the dictionary
	- $y^{\langle t\rangle}$ : ground-truth, one-hot-encoding with $K$ classes
	- Minimise the cross-entropy loss
$$
\mathscr{L}=\sum_t \mathscr{L}\left(\hat{y}^{\langle t\rangle}, y^{\langle t\rangle}\right) \quad \text { with } \quad \mathscr{L}\left(\hat{y}^{\langle t\rangle}, y^{\langle t\rangle}\right)=-\sum_{c=1}^K y_c^{\langle t\rangle} \log \left(\hat{y}_c^{\langle t\rangle}\right)
$$
- **Testing**
	- Given a new sentence of three words $y^{\langle 1\rangle}, y^{\langle 2\rangle}, y^{\langle 3\rangle}$ what is its probability?
$$
\begin{aligned}
p\left(y^{\langle 1\rangle}, y^{\langle 2\rangle}, y^{\langle 3\rangle}\right)= & p\left(y^{\langle 1\rangle}\right) & & \text { given by the first softmax } \\
& p\left(y^{\langle 2\rangle} \mid y^{\langle 1\rangle}\right) & & \text { given by the second softmax } \\
& p\left(y^{\langle 3\rangle} \mid y^{\langle 1\rangle}, y^{\langle 2\rangle}\right) & & \text { given by the third softmax }
\end{aligned}
$$
###### Sampling novel sequences using RNN
![alt text](/Photos/Screenshot%202023-10-05%20at%2009.40.04.png)
<!-- ![[../Photos/Screenshot 2023-10-05 at 09.40.04.png]] -->
So, here basically we use the shared weight from the training to generate a novel sequence.

##### One-Hot Encoding

Question: how to represent the individual words in a sentence?
	- What will be $x^{\langle t \rangle}$ ?
Construct the list of the $N$ most used categories (words) in the corpus.
	- "vocabulary/dictionary"
	- $N$ is usually in the range 10.000 words 50.000 .
	- add a $<$ UNK  (unknown word) for the words that are not in vocabulary/dictionary.
Each word is now associated with an ID.


| 1 | Angela |
| :---: | :---: |
| $\ldots$ | Boris |
| $\ldots$ | chancelor |
| 367 | Emmanuel |
| $\ldots$ | France |
| $\ldots$ | Germany |
| 4075 | is |
| $\ldots$ | Johnson |
| $6830$ | Macron |
| $\ldots$ | Merkel |
| $\ldots$ | of |
| $\ldots$ | president |
| $\cdots$ | prime-minister |
| $\cdots$ | the |
| 10.000 | UK |

- **One-hot-encoding:**
	- the one-hot vector of an ID is a vector filled with 0s, except for a 1 at the position associated with the ID
	- each word $w$ is associated with a one-hot-vector vectoro $o_{I D}$
	- If $N=10.000, x^{\langle t\rangle}$ has dimension 10.000
- a one-hot encoding makes no assumption about word similarity
	- all words are equally different from each other

| $\{x\}$ | $x^{<1}$ | $x^{<2}$ | $x^{<3}$ | $x^{<4}$ | $x^{\langle 5\rangle}$ | $x^{\langle 6\rangle}$ | $x^{\langle 7\rangle}$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|  | Emmanuel | Macron | is | the | president | of | France |
| 1 | 0 | 0 | 0 |  |  |  |  |
| 2 | 0 | 0 | 0 |  |  |  |  |
| $\ldots$ | $\ldots$ | $\ldots$ | $\ldots$ |  |  |  |  |
| 367 | 1 | 0 | 0 |  |  |  |  |
| $\ldots$ | $\ldots$ | $\ldots$ | $\ldots$ |  |  |  |  |
| 4075 | 0 | 0 | 1 |  |  |  |  |
| $\ldots$ | $\ldots$ | $\ldots$ | $\ldots$ |  |  |  |  |
| 6830 | 0 | 1 | 0 |  |  |  |  |
| $\ldots$ | $\ldots$ | $\ldots$ | $\ldots$ |  |  |  |  |
| 10.000 | 0 | 0 | 0 |  |  |  |  |

**Weaknesses of one-hot-encoding**
- it is very high-dimensional
	- vulnerability to overfitting, computationally expensive
- all words are equally different from each other
	- the inner product between any two one-hot vectors is zero
	- as a consequence, we cannot generalise
- Example:
	- we have a trained a language model to complete the sentence
		- The president is on a boat" _ ? _ $\Rightarrow$ "trip"
	- since there is no specific relationship between "boat" and "plane", the algorithm cannot complete the sentence
		- The president is on a plane _ ? _  $\Rightarrow$ ?
##### Word embedding
---
- learn a continuous representation of words
- each word wis associated with a real-valued vector $e_{I D}$
- if embedding size is $300, e_{5391}$ is a 300 dimensional vector
- we would like the distance $\left\|e_{I D 1}-e_{I D 2}\right\|$ to reflect the meaningful similarities between words

|  | Man | Woman | King | Queen | Uncle | Aunt | President | Chancelor | France | Germany | Boat | Plane |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|  | 5391 | 9853 | 4914 | 7157 | 456 | 6257 | 7124 | 3212 | 6789 | 1234 | 5923 | 8871 |
| Gender | -1 | 1 | -1 | 1 | -1 | 1 | $-0,7$ | $-0,3$ | 0 | 0 | 0 | 0 |
| Royal | 0 | 0 | 1 | 1 | 0 | 0 | 0,5 | 0,5 | 0 | 0 | 0 | 0 |
| Age | 0 | 0 | 0,7 | 0,7 | 0,5 | 0,5 | 0,7 | 0,7 | 0 | 0 | 0 | 0 |
| Country | 0 | 0 | 0,5 | 0,5 | 0 | 0 | 0,5 | 0,2 | 1 | 1 | 0 | 0 |
| Transportation | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| $\cdots$   | $\cdots$ |   $\cdots$ | $\cdots$ |  $\cdots$  |  $\cdots$  | $\cdots$ |   $\cdots$ |  $\cdots$  | $\cdots$ |  $\cdots$  | $\cdots$ |   $\cdots$ |

- When we represent like this, "boat" and "plane" are quiet similar.
	- Some of the features may differs but most features are similar. 
	- We can predict, boat and plane in the above example.
##### Word embedding / Matrix
---
![alt text](/Photos/Screenshot%202023-10-05%20at%2010.10.09.png)
<!-- ![[../Photos/Screenshot 2023-10-05 at 10.10.09.png |center]] -->

##### Word embedding/ Classic neural language model
---
- **Method**: build a language model (learn to predict the following word) using a MLP
	- Use back-propagation to learn the parameters: $\mathbf{E}, \mathbf{W}^{[1]}, \mathbf{b}^{[1]}, \mathbf{W}^{[2]}, \mathbf{b}^{[2]}$
	- $\mathbf{E}$ is the embedding matrix
![alt text](/Photos/Screenshot%202023-10-05%20at%2010.12.15.png)
 <!-- ![[../Photos/Screenshot 2023-10-05 at 10.12.15.png |center]] -->

##### Application: Sentiment Classification / Using a simple model
---
![alt text](/Photos/Screenshot%202023-10-05%20at%2010.53.50.png)
<!-- ![[../Photos/Screenshot 2023-10-05 at 10.13.42.png |center]] -->

**Does not work for**
"This movie was completely **lacking** a good script, good actors and a good director"
- This is because the Avg of "good" is positive
- The model can not not understand that we means the opposite ("lacking")
- We need a Sequence model $\Rightarrow R N N$

![alt text](/Photos/Screenshot%202023-10-05%20at%2010.15.03.png)
<!-- ![[../Photos/Screenshot 2023-10-05 at 10.15.03.png]] -->

## Sequence to sequence [Sutskever et al., 2014] or Encoder-Decoder [Cho et al., 2014]
---

![alt text](/Photos/Screenshot%202023-10-05%20at%2010.17.00.png)
<!-- ![[../Photos/Screenshot 2023-10-05 at 10.17.00.png]] -->
seq2seq and encoder-decoder is similar, encoder-decoder uses architecture of seq2seq but it uses a context vector. The context vector is a weighted sum of the encoder's hidden states, and it is used to provide the decoder with information about the entire input sequence. This allows the decoder to generate more accurate and informative output sequences.

**Example: Machine Translation**
- Language model
	- $p\left(y^{\langle 1\rangle}, y^{\langle 2\rangle}, \ldots, y^{\left\langle T_y\right\rangle}\right)$
- **Machine Translation:**
	- Conditional language model: $p\left(y^{\langle 1\rangle}, \ldots, y^{\left\langle T_y\right\rangle} \mid x^{\langle 1\rangle}, \ldots, x^{\left\langle T_x\right\rangle}\right)$
	- English sentence: $x^{\langle 1\rangle}, \ldots, x^{\left\langle T_x\right\rangle}$
	- French sentence: $y^{\langle 1\rangle}, \ldots, y^{\left\langle T_y\right\rangle}$
![alt text](/Photos/Screenshot%202023-10-05%20at%2010.28.01.png)
<!-- ![[../Photos/Screenshot 2023-10-05 at 10.28.01.png]] -->
## Neural Machine Translation (NMT) (Standford Slides)
---
![alt text](/Photos/Screenshot%202023-10-12%20at%2012.02.12.png)
<!-- ![[../Photos/Screenshot 2023-10-12 at 12.02.12.png |center| 500]] -->
- The general notion here is an **encoder-decoder** model
	- One neural network takes input and produces a neural representation
	- Another network produces output based on that neural representation
	- If the input and output are sequences, we call it a seq2seq model
- **Sequence-to-sequence** is useful for **more than just MT**
- Many NLP tasks can be phrased as sequence-to-sequence:
	- **Summarization** (long text $\rightarrow$ short text)
	- **Dialogue** (previous utterances $\rightarrow$ next utterance)
	- **Parsing** (input text $\rightarrow$ output parse as sequence)
	- **Code generation** (natural language $\rightarrow$ Python code)

## Attention Model
---
#### Problem of long sequences
---
![alt text](/Photos/Screenshot%202023-10-05%20at%2010.29.53.png)
<!-- ![[../Photos/Screenshot 2023-10-05 at 10.29.53.png]] -->
- **Encoder/Decoder:**
	- $a^{\left\langle T_x\right\rangle}$ is supposed to memorise the whole sentence then translate it;
	- human way: translate each part of a sentence at a time
- **Attention model?** (modification of the Encoder/Decoder)
	- $a^{\left\langle T_x\right\rangle}$ is replaced by a local version in the encoder $\Rightarrow$ we pay attention on specific encoding!
![alt text](/Photos/Screenshot%202023-10-05%20at%2010.32.55.png)
<!-- ![[../Photos/Screenshot 2023-10-05 at 10.32.55.png]] -->
- In attention model:
	- each decoding time $\tau$ has its own context: $c^{\langle\tau\rangle}$
	- the context $c^{\langle\tau\rangle}$ is computed as a weighted sum of the encoding hidden states $a^{\langle t\rangle}$
- weights $\alpha^{\langle\tau, t\rangle}$
	- attention weights
	- when generating information at time $\tau$, how much, do we have to pay attention on information at time $t=\{1,2,3 \ldots, 7\}$
- Attention weights $\alpha^{\langle\tau=3, t\rangle}$ describe an "alignment" between information at
	- **encoding time $t$ :**
		- computed using $\vec{a}^{\langle t\rangle}$ and $\overleftarrow{a}^{\langle t\rangle} ;$ we note $a^{\langle t\rangle}=\left[\vec{a}^{\langle t\rangle}, \overleftarrow{a}^{\langle t\rangle}\right]$
	- **decoding time $\tau=3$ :**
		- computed using $s^{\langle\tau=2\rangle}$ (we do not yet observe $\tau=3$ )
- The context vector $c^{\langle\tau\rangle}$ at decoding time $\tau$
	- computed as the sum of the annotations $a^{\langle t\rangle}$ weighted by their attention weights $\alpha^{\langle\tau, t\rangle}$
$$
c^{\langle\tau\rangle}=\sum_t \alpha^{\langle\tau, t\rangle} a^{\langle t\rangle}
$$
- The attention weight $\alpha^{\langle\tau, t\rangle}$
	- $\alpha^{\langle\tau, t\rangle}=$ represents the amount of attention $y^{\langle\tau\rangle}$ should pay to $a^{\langle t\rangle}$
	- computed using a softmax on the alignment model $e^{\langle\tau, t\rangle}$
$$
\alpha^{\langle\tau, t\rangle}=\frac{\exp \left(e^{\langle\tau, t\rangle}\right)}{\sum_{t^{\prime}=1}^{T_x} \exp \left(e^{\left\langle\tau, t^{\prime}\right\rangle}\right)} \text { with } \sum_t \alpha^{\langle\tau, t\rangle}=1
$$
- The alignment model $e^{\langle\tau, t\rangle}$ scores how well the inputs around position $t$ match the output at position $\tau$
	- is computed using two inputs
		- **query**: the RNN hidden state $s^{\langle\tau-1\rangle}$ of the output sequence (just before emitting $y^{\langle\tau\rangle}$ )
		- **keys**: the RNN hidden states $a^{\langle t\rangle}$ of the input sentence.
	- is computed using $f\left(s^{\langle\tau-1\rangle}, a^{\langle t\rangle}\right)$
		- can be a feedforward neural network (jointly trained with all the other components)
		- can be a dot-product: $e^{\langle t\rangle}=s^T a^{\langle t\rangle}$
		- can be multiplicative attention: $e^{\langle t\rangle}=s^T W a^{\langle t\rangle}$
		- In this case, the hidden state $a^{\langle t\rangle}$ is the key $k^{\langle t\rangle}$![[../Photos/Screenshot 2023-10-05 at 10.43.48.png]]
## Transformer
---

### Self-Attention
---
- Self-Attention?
	- As the model processes each word (each position in the input sequence), self attention allows the process to look at other positions in the sequence for clues that can help lead to a better encoding for this word
	- Example: when the model is processing the word "it", self-attention allows it to associate "it" with "animal"
- Difference between self-attention and attention is that:
	- Attention allows the process to look at the specific position in the sequence
	- self-attention allows the process to look at any position in the sequence.
- ![alt text](/Photos/Screenshot%202023-10-05%20at%2010.47.59.png)
- ![alt text](/Photos/Screenshot%202023-10-05%20at%2010.48.27.png)
<!-- - ![[../Photos/Screenshot 2023-10-05 at 10.47.59.png]]
- ![[../Photos/Screenshot 2023-10-05 at 10.48.27.png]] -->
##### Implementation:
---
1. First step :
- create three vectors from each of the encoder's input vectors $x_i$ :
	- a Query vector $q_i$,
	- a Key vector $k_i$
	- a Value vector $v_i$
- the vectors are created by multiplying the input by three matrices $W^Q, W^K, W^V$ that are trained during the training process
- the new vectors are smaller in dimension than the embedding vector
- ![alt text](/Photos/Screenshot%202023-10-05%20at%2010.53.50.png)
<!-- - ![[../Photos/Screenshot 2023-10-05 at 10.53.50.png]] -->
![alt text](/Photos/Screenshot%202023-10-05%20at%2010.49.02.png)
<!-- ![[../Photos/Screenshot 2023-10-05 at 10.49.02.png | 400]] -->
2. **Second step : calculate a score**
- The score determines how much focus to place on other parts of the input sentence as we encode a word at a certain position.
- Example : calculate self-attention for the first word "Thinking" $\Rightarrow$ we score each word of the input sentence against this word
- Score?
	- dot product of the query vector $q_i$ with the key vector $k_i$ of the respective word we're scoring
	- divide by 8 (the square root of the dimension of the key vectors - 64)
	- pass the result through a softmax operation
	- softmax score determines how much each word will be expressed at this position.
3. **Third step :**
- multiply each value $v_i$ vector by the softmax score
	- keep intact the values of the word(s) we want to focus on, and drown-out irrelevant words
- sum up the weighted value vectors
	- produces the output $z_i$ of the self-attention layer at this position (for the first word).
![alt text](/Photos/Screenshot%202023-10-05%20at%2010.56.50.png)

<!-- ![[../Photos/Screenshot 2023-10-05 at 10.56.50.png]] -->

#### Transformer Architecture
---
![alt text](/Photos/Screenshot%202023-10-05%20at%2010.59.34.png)
<!-- ![[../Photos/Screenshot 2023-10-05 at 10.59.34.png]] -->
- Bottom input
	- each input word is turned into a vector using an embedding algorithm (only happens in the bottom-most encoder)
- Each encoder block:
	- receives a list of vectors as input
	- processes this list by
		- passing these vectors into a self-attention layer,
			- represents the dependencies between the inputs
		- then into a feed-forward neural network,
			- does not have those dependencies and thus can be process the various steps in parallel while flowing through the feed-forward layer
		- then sends out the output upwards to the next encoder block
			
	![alt text](/Photos/Screenshot%202023-10-05%20at%2011.01.13.png)
	<!-- ![[../Photos/Screenshot 2023-10-05 at 11.01.13.png]] -->
**- Residuals + Normalisation**
	- each sub-layer (self-attention, feedforward) in each encoder has
		- a residual connection around it
		- is followed by a layer-normalisation step
		
- Layer norm is particularly important in the Transformer architecture because it helps to prevent the gradients from exploding or vanishing. This is because the Transformer architecture uses a self-attention mechanism, which can cause the gradients to become very large or very small. Layer norm helps to keep the gradients within a reasonable range, which makes the training process more stable.
- Residual connection adds the input of a layer to its output. This allows the layer to learn the difference between the input and the output, rather than trying to learn the entire function from scratch.


- The decoder has
	- (as the encoder) both Self-Attention and Feed Forward layers
	- but also an Attention layer between the encoder and decoder
		- helps the decoder focus on relevant parts of the input sentence
		- similar what attention does in Seq2Seq models
	![alt text](/Photos/Screenshot%202023-10-05%20at%2011.10.00.png)
	<!-- ![[../Photos/Screenshot 2023-10-05 at 11.10.00.png]] -->

- Decoder side
	- The output of the top encoder is then transformed into a set of attention vectors $\mathrm{K}$ and $\mathrm{V}$
		- to be used by each decoder in its "encoder-decoder attention" layer which helps the decoder focus on appropriate places in the input sequence
	- The output of the decoder of each step is fed to the bottom in the next step
	- As for the encoder, we embed and add positional encoding to those decoder inputs
	- In the decoder, the self-attention layer is only allowed to attend to earlier positions in the output sequence
	- The "Encoder-Decoder Attention" layer works just like multi-head self-attention,
		- except it creates its **Queries matrix from the layer below it**, and takes **the Keys and Values matrix** from the output of the encoder stack.
- Representing The Order of The Sequence Using Positional Encoding
	- Find a way to account for the order of the words in the input sequence
		- Transformer adds a vector to each input embedding
		- These vectors follow a specific pattern that the model learns, which helps it determine the position of each word, or the distance between different words in the sequence
		![alt text](/Photos/Screenshot%202023-10-05%20at%2011.12.32.png)
		<!-- ![[../Photos/Screenshot 2023-10-05 at 11.12.32.png]] -->

