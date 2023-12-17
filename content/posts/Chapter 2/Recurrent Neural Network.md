---
Topic: Recurrent Neural Network
Cours: IA306 Deep Learning
Date: 28-09-2023
tags:
  - Deep_learning
---


![[Deep Learning IA306/Chapter 2/Slides/Page2.png]]
Fully-Connected is really special in this case. All the algorithms about SGD, Adam, ... can be used in every case, even they are not fully connected.
![[Deep Learning IA306/Chapter 2/Slides/Page3.png]]
It is stupid if we consider that the sequential data is independent. 
Attention, transformer takes benefit from the sequentiality of the data.

![[Deep Learning IA306/Chapter 2/Slides/Page4.png]]

![[Deep Learning IA306/Chapter 2/Slides/Page5.png]]
What are sequential data? 
- Sequential data are almost every data that we can imagine, most of the time, they are time series data, fourier transforms, ...

What is vector linear autoregressive model? 
- A vector linear autoregressive (VAR) model is a statistical model that predicts future values of a set of variables based on their own past values and the past values of the other variables in the model. VAR models are a multivariate extension of the autoregressive (AR) model, which is a univariate model that predicts future values of a single variable based on its own past values.
What is Markov Model?
- A Markov model is a mathematical model that describes a sequence of events in which the probability of each event depends only on the state attained in the previous event
- 

![[Deep Learning IA306/Chapter 2/Slides/Page6.png]]
For example, the machine translation the sequence of inputs of $T_x$ and $T_y$ are different. That's why RNN is powerful.


![[Deep Learning IA306/Chapter 2/Slides/Page7.png]]

| type              | activation | loss |
| ----------------- | ---------- | ---- |
| Continuous values | Linear     | MSE  |
| Binary-classes     | Sigmoid    | BCE  |
| Multi-classes       | Softmax    | CE   |


How do we process categorical values $x^{<t>} \in \{1,\cdots,K\}$  as input ?
- One-hot encoding: transform the categorical values to 0 or 1.
- Embedding: convert the input into an array of values. 
- 
![[Deep Learning IA306/Chapter 2/Slides/Page8.png]]
$T_x$ can be different at each time -> the dimension would be different for each $i$.
Solution: 
- Zero-padding?
- It would be still requires a huge input dimension

**Problem:** No weight sharing, which means in each position our NN has to learn and train the weight.
Example: 
- Emmanuel .... -> learn for 1st position
- Currently Emmanuel .... -> learn for 2nd position


![[Deep Learning IA306/Chapter 2/Slides/Page9.png]]

![[Deep Learning IA306/Chapter 2/Slides/Page10.png]]

![[Deep Learning IA306/Chapter 2/Slides/Page11.png]]

![[Deep Learning IA306/Chapter 2/Slides/Page12.png]]

![[Deep Learning IA306/Chapter 2/Slides/Page13.png]]
Process 2nd data, get some knowledge from the process of 1st data.
a, is the hidden state, value of projection of each state.
![[Deep Learning IA306/Chapter 2/Slides/Page14.png]]
Process at time 3, we get info from time 2 and time 1
![[Deep Learning IA306/Chapter 2/Slides/Page15.png]]

![[Deep Learning IA306/Chapter 2/Slides/Page16.png]]

![[Deep Learning IA306/Chapter 2/Slides/Page17.png]]

![[Deep Learning IA306/Chapter 2/Slides/Page18.png]]

![[Deep Learning IA306/Chapter 2/Slides/Page19.png]]

![[Deep Learning IA306/Chapter 2/Slides/Page20.png]]

![[Deep Learning IA306/Chapter 2/Slides/Page21.png]]

![[Deep Learning IA306/Chapter 2/Slides/Page22.png]]

![[Deep Learning IA306/Chapter 2/Slides/Page23.png]]
$g_2$ is the activation function to get the output. 
![[Deep Learning IA306/Chapter 2/Slides/Page24.png]]

![[Deep Learning IA306/Chapter 2/Slides/Page25.png]]

![[Deep Learning IA306/Chapter 2/Slides/Page26.png]]

![[Deep Learning IA306/Chapter 2/Slides/Page27.png]]

![[Deep Learning IA306/Chapter 2/Slides/Page28.png]]

![[Deep Learning IA306/Chapter 2/Slides/Page29.png]]

![[Deep Learning IA306/Chapter 2/Slides/Page30.png]]

![[Deep Learning IA306/Chapter 2/Slides/Page31.png]]

![[Page32.png]]

![[Page33.png]]

![[Page34.png]]
Is the result always the same with different path?
![[Page35.png]]
When we do the computation, we like to do parallelism. So RNN is slow and is not parallelisable.
![[Page36.png]]

![[Page37.png]]
What if our task is like classification or sentiment analysis, how can we define a lost?
![[Page38.png]]

![[Page39.png]]

![[Page40.png]]

![[Page41.png]]

![[Page42.png]]

![[Page43.png]]

![[Page44.png]]

![[Page45.png]]

![[Page46.png]]

![[Page47.png]]

![[Page48.png]]

![[Page49.png]]

![[Page50.png]]
gamma equal 1 = keep in memory

|Feature|GRU|LSTM|
|---|---|---|
|Number of gates|2|3|
|Gates|Reset gate, update gate|Input gate, forget gate, output gate|
|Cell state|No|Yes|
|Complexity|Lower|Higher|
| Accuracy | Lower | Higher |



![[Page51.png]]

![[Page52.png]]

![[Page53.png]]

![[Page54.png]]

![[Page55.png]]

![[Page56.png]]

![[Page57.png]]

![[Page58.png]]

![[Page59.png]]

![[Page60.png]]

![[Page61.png]]

![[Page62.png]]

![[Page63.png]]

![[Page64.png]]

![[Page65.png]]

![[Page66.png]]


![[Page67.png]]

![[Page68.png]]

![[Page69.png]]

![[Page70.png]]

![[Page71.png]]

![[Page72.png]]

![[Page73.png]]

![[Page74.png]]

![[Page75.png]]

![[Page76.png]]

![[Page77.png]]

![[Page78.png]]

![[Page79.png]]

![[Page80.png]]

![[Page81.png]]
seq2seq = take an RNN as an input
encoder-decoder = decoder 
![[Page82.png]]

![[Page83.png]]

![[Page84.png]]

![[Page85.png]]

![[Page86.png]]

![[Page87.png]]

![[Page88.png]]

![[Page89.png]]

![[Page90.png]]

![[Page91.png]]



Attention lets us choose the right decoder. 
![[Page92.png]]

![[Page93.png]]

![[Page94.png]]

![[Page95.png]]

![[Page96.png]]

![[Page97.png]]

![[Page98.png]]

![[Page99.png]]

![[Page100.png]]

![[Page101.png]]

![[Page102.png]]

![[Page103.png]]

![[Page104.png]]

![[Page105.png]]

![[Page106.png]]

![[Page107.png]]

![[Page108.png]]

![[Page109.png]]

![[Page110.png]]

![[Page111.png]]

![[Page112.png]]

![[Page113.png]]

![[Page114.png]]

![[Page115.png]]

![[Page116.png]]

![[Page117.png]]

![[Page118.png]]

![[Page119.png]]

![[Page120.png]]

![[Page121.png]]

![[Page122.png]]

![[Page123.png]]

![[Page124.png]]

![[Page125.png]]