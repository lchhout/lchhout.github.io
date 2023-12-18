---
title: "Multi-Layers Perceptron"
date: 2023-12-17T17:21:05+01:00
# bookComments: false
# bookSearchExclude: false
math: katex
markup:
  defaultMarkdownHandler: goldmark
# tags:
#   - Deep Learning
author: lchhout99

---
## Introduction
---
- Machine Learning lets the machine to learn from the data by applying models such as XGBoost, Random Tree, Linear Regression, clustering, etc.
- Deep learning lets the machine to learn from the data without knowing much about the data structure or domain knowledge.
### Learning Representation
---

- **Rule-based systems**: Input -> Hand-designed program -> Output (Symbolic AI)
- **Classical ML**: Input -> Hand-designed features -> mapping from features (Models) -> Output
- **Representation Learning**:
	- Input -> Features -> Mapping from the Features (Model) -> Output \[Without knowing the importance of the features or domain knowledge]
	- Input -> Simple features -> Multi-Layer perceptron (Additional layers of more abstract features) -> Mapping from features -> Output

The idea behind deep learning is that we take an input and project it into a **lower-dimensional space** which will capture a lot of input detail (pixels) and all those input details are **orthogonal** and **maximize the variance**. 
- **Lower-dimensional space**: Deep learning projects inputs into a lower-dimensional space because a lower-dimensional space means that the representation of the input is more concise and efficient, without losing too much information.
- **Orthogonal:** Deep learning tries to project inputs into a space where the dimensions are orthogonal, meaning that they are independent of each other. This makes it easier for the model to learn the different features of the input.
- **Maximize the variance:** Deep learning tries to project inputs into a space where the variance is maximized. This means that the different dimensions of the representation contain as much information as possible about the input.
#### Perceptron
---
**Perceptron** is a simple machine learning algorithm that can be used to train linear models. However, perceptron is limited in that it can only separate linearly separable data -> if the data points cannot be divided into two classes by a straight line or hyperplane, then perceptron will not be able to learn to classify them correctly. 
**Deep learning** overcomes this limitation by using multiple layers of perceptrons. 
- The first layer of perceptrons learns to extract simple features from the input data. 
- The second layer of perceptrons learns to combine these simple features to extract more complex features. And so on. 
By using multiple layers of perceptrons, deep learning models can learn to represent and classify even very complex data sets.
**Deep learning** is said to be a projection of non-linearity because it uses a series of layers to learn non-linear relationships between the input and output of a model. Each layer in a deep learning model is a linear transformation, but the combination of multiple layers can learn non-linear relationships.

```
Example: Imagine we have a dataset of images of cats and dogs. Perceptron would not be able to classify these images correctly, because the data is not linearly separable. However, a deep learning model would be able to learn to classify these images correctly by extracting complex features from the images, such as the shape of the ears, the color of the fur, and the position of the eyes. These features are non-linearly related to the input image, but the deep learning model can learn to represent these relationships by using a series of linear transformations.
```

**Manifold:** Deep learning represent an input in a subspace and not in the real space, after that do the projection to find the manifold.
```
Manifold learning is a way of finding the underlying structure of high-dimensional data by projecting it onto a lower-dimensional subspace. The subspace is chosen such that it preserves the essential features of the data, such as its topological properties.
	
The first step in manifold learning is to represent the input data in a subspace. This can be done by using a variety of techniques, such as principal component analysis (PCA) or kernel principal component analysis (KPCA). PCA finds the directions in which the data varies the most, and projects the data onto these directions. KPCA is a generalization of PCA that can be used to find non-linearly separable data.
	
Once the data has been represented in a subspace, the next step is to find the manifold. This can be done by using a variety of techniques, such as locally linear embedding (LLE) or isometric feature mapping (Isomap). LLE finds the neighbors of each data point and then fits a linear model to the neighbors. Isomap finds the shortest paths between all pairs of data points and then embeds the data into a lower-dimensional space such that the shortest paths are preserved.
	
The manifold learning technique described above is just one example of how deep learning can be used to represent data in a subspace and then find the underlying structure of the data. There are many other techniques that can be used, and the choice of technique depends on the specific data set and the task at hand.
```
	
## Chapter 1 : Multi-Layer Perceptrons
--- 

![alt text](/MLP/Page1.jpg)
<!-- ![[Deep Learning IA306/Chapter 1/Page1.jpg]] -->


![alt text](/MLP/Page1.jpg)
<!-- ![[Deep Learning IA306/Chapter 1/Page2.jpg]] -->
Modify Perceptron -> Logistic Regression -> Neural Network (Combine) -> Deep Neural Networks
Non-linearity activation function that we used in deep learning are: sigmoid, tanh, relu, but some of them can lead to vanishing gradient. 
```note-purple
Vanishing gradient happend when the derivative is to small.
```
### Perceptron Models
---
![alt text](/MLP/Page6.jpg)
<!-- ![[Deep Learning IA306/Chapter 1/Page6.jpg]] -->
The problem is that we have to know the good projection or connection which is $w$ and the $b$. 

![alt text](/MLP/Page7.jpg)
<!-- ![[Deep Learning IA306/Chapter 1/Page7.jpg]] -->
$X$ is the input and $y$ is the ground truth. 
**The update step in algorithm is to modify something because our system is wrong (we update the weight).**
This is equivalent to minimising a **loss function.**
![alt text](/MLP/Page8.jpg)
<!-- ![[Deep Learning IA306/Chapter 1/Page8.jpg]] -->
We see that the threshold function is not derivative at $0$ which leads to instability during training.
![alt text](/MLP/Page10.jpg)
<!-- ![[Deep Learning IA306/Chapter 1/Page10.jpg]] -->
A neural network is a function $f_{\theta}(X)$ which gives a prediction $\hat{y}$ of a ground-truth value $y$. If there is an error, we create a risk. So the idea is, we need to find a system $f_\theta$ that will create a reality.
![alt text](/MLP/Page11.jpg)
<!-- ![[Deep Learning IA306/Chapter 1/Page11.jpg]] -->
Study what is the relationship between the loss $\mathscr{L}_\theta$ and $\theta$ of the system. 
At initialisation, we only compute the lost based on the $w_1$ and after that we will update the weight by using gradient because we know that it will increase the lost, so we move **at opposite direction of the gradient.**
But in order to use gradient descend we need $\mathscr{L}$ that is differentiable, but the threshold is not differentiate at $0$. 
![alt text](/MLP/Page12.jpg)
<!-- ![[Deep Learning IA306/Chapter 1/Page12.jpg]] -->
- If the loss function $\mathscr{L}$ is non-convex function, there is no global minima so we need to use more elaborate optimisation algorithm. 
- What if there are 1000 layers, how can we compute the derivative ? -> **Back-propagation algorithm.**
![alt text](/MLP/Page13.jpg)
<!-- ![[Deep Learning IA306/Chapter 1/Page13.jpg]] -->
Sigmoid jump into scene because it is differentiable at all point.
![alt text](/MLP/Page14.jpg)
<!-- ![[Deep Learning IA306/Chapter 1/Page14.jpg]] -->
But why do we choose binary-cross-entropy (BCE) as our loss in this case?
The binary classification requires us to use sigmoid and we will always use BCE.
![alt text](/MLP/Page16.jpg)
<!-- ![[Deep Learning IA306/Chapter 1/Page16.jpg]] -->
![alt text](/MLP/Page17.jpg)
<!-- ![[Deep Learning IA306/Chapter 1/Page17.jpg]] -->
Logistic regression predict the output by modelling the log-odds of the outcome as a linear function of the input variables. 
![alt text](/MLP/Page18.jpg)
<!-- ![[Deep Learning IA306/Chapter 1/Page18.jpg]] -->
We see that maximise log-likelihood of the $y$ given the input $x$ is to minimise a loss named Binary Cross-Entropy (BCE). That's why binary classification requires BCE.
Bard's Answer: [when we train a binary classification model using the BCE loss, we are essentially maximizing the log-likelihood of the target variable given the input variable. This helps us to train a model that can accurately predict the probability of the target variable for any given input.]
![alt text](/MLP/Page19.jpg)
<!-- ![[Deep Learning IA306/Chapter 1/Page19.jpg]] -->


### Logistic Regression
---
Logistic Regression is 0 hidden layers because we observe the input and output directly.
![alt text](/MLP/Page21.jpg)
<!-- ![[Deep Learning IA306/Chapter 1/Page21.jpg]] -->
![alt text](/MLP/Page22.jpg)
<!-- ![[Deep Learning IA306/Chapter 1/Page22.jpg]] -->
Binary-Classification -> Binary-Cross-Entropy 
In general, when we deal with neural network we can have several lost fonctions.

### Gradient Descent
---
![alt text](/MLP/Page23.jpg)
<!-- ![[Deep Learning IA306/Chapter 1/Page23.jpg]] -->
We move by applying the learning rate, which is small step. How to choose it? It is a big problem. 
- If the learning rate is too large, the model may skip over the minimum and not converge to the optimal solution. This is because the model will take large steps in the parameter space, which may cause it to miss the minimum altogether.
- If the learning rate is too small, the model will take many steps to converge to the optimal solution. This can be time-consuming, especially for large and complex models.
There is no **one-size-fits-all** answer to the question of how to choose the learning rate. The best learning rate will vary depending on the specific machine learning task and the model being used. However, there are a few general guidelines that can be followed:
- Start with a small learning rate and gradually increase it until the model starts to overfit the training data. Then, reduce the learning rate slightly until the model is able to generalize well to unseen data.
- Use a learning rate scheduler to adjust the learning rate during training. This can help to improve the performance of the model and speed up the training process.
- Track the training loss and accuracy metrics to monitor the performance of the model. If the loss starts to increase rapidly, or if the accuracy starts to decrease, then the learning rate may be too high.
![alt text](/MLP/Page24.jpg)
<!-- ![[Deep Learning IA306/Chapter 1/Page24.jpg]] -->
![alt text](/MLP/Page25.jpg)
<!-- ![[Deep Learning IA306/Chapter 1/Page25.jpg]] -->
![alt text](/MLP/Page26.jpg)
<!-- ![[Deep Learning IA306/Chapter 1/Page26.jpg]] -->
![alt text](/MLP/Page27.jpg)
<!-- ![[Deep Learning IA306/Chapter 1/Page27.jpg]] -->
A backward propagation is just a way to compute the gradient of the function, it is efficient and easy to implement.

### Chain-rule
---
![alt text](/MLP/Page29.jpg)
<!-- ![[Deep Learning IA306/Chapter 1/Page29.jpg]] -->
**Back-propagation is an efficient algorithm to compute the chain-rule by storing intermediate (and re-use derivatives)**
![alt text](/MLP/Page30.jpg)
<!-- ![[Deep Learning IA306/Chapter 1/Page30.jpg]] -->
![alt text](/MLP/Page31.jpg)
<!-- ![[Deep Learning IA306/Chapter 1/Page31.jpg]] -->
Basically, when we compute direct derivative we calculate in the go-direction and if we compute using back-propagation, we compute in the opposite direction.

![alt text](/MLP/Page32.jpg)
<!-- ![[Deep Learning IA306/Chapter 1/Page32.jpg]] -->
![alt text](/MLP/Page33.jpg)
<!-- ![[Deep Learning IA306/Chapter 1/Page33.jpg]] -->
![alt text](/MLP/Page34.jpg)
<!-- ![[Deep Learning IA306/Chapter 1/Page34.jpg]] -->
![alt text](/MLP/Page35.jpg)
<!-- ![[Deep Learning IA306/Chapter 1/Page35.jpg]] -->
#### Numpy code to calculate the back-propagation
![alt text](/MLP/Page36.jpg)
<!-- ![[Deep Learning IA306/Chapter 1/Page36.jpg]] -->
![alt text](/MLP/Page37.jpg)
<!-- ![[Deep Learning IA306/Chapter 1/Page37.jpg]] -->
The XOR operator is not a linear separator. This means that there is no hyperplane that can be used to divide a set of points that satisfy the XOR operator into two classes.

For example, consider the following set of points:

```
(0, 0)
(1, 0)
(0, 1)
(1, 1)
```

These points satisfy the XOR operator, because the XOR of 0 and 0 is 0, the XOR of 1 and 0 is 1, the XOR of 0 and 1 is 1, and the XOR of 1 and 1 is 0.
![alt text](/MLP/Page38.jpg)
<!-- ![[Deep Learning IA306/Chapter 1/Page38.jpg]] -->
![alt text](/MLP/Page39.jpg)
<!-- ![[Deep Learning IA306/Chapter 1/Page39.jpg]] -->
![alt text](/MLP/Page40.jpg)
<!-- ![[Deep Learning IA306/Chapter 1/Page40.jpg]] -->
The hidden state projects the input data into a new subspace which make our data linear.
In the example above, the hidden state divide our input data which is: 
```
(0, 0)
(1, 0)
(0, 1)
(1, 1)
```
into 
```
(0)
(1)
(0)
(1)
and 
(0)
(0)
(1)
(1)
```

![alt text](/MLP/Page1.jpg)
<!-- ![[Deep Learning IA306/Chapter 1/Page41.jpg]] -->
A Neural Network with only one hidden layer can approximate any function (with enough hidden neurons)
https://medium.com/analytics-vidhya/neural-networks-and-the-universal-approximation-theorem-e5c387982eed
```
The reason why neural networks are universal approximators is because they can learn to represent complex non-linear relationships between the input and output variables. This is because neural networks are made up of interconnected nodes, called neurons, which process information and transmit it to other neurons. Each neuron applies a non-linear activation function to its input, which allows the network to learn to represent complex non-linear relationships.
```
![alt text](/MLP/Page42.jpg)
<!-- ![[Deep Learning IA306/Chapter 1/Page42.jpg]] -->
### Neural Networks
---
![alt text](/MLP/Page44.jpg)
<!-- ![[Deep Learning IA306/Chapter 1/Page44.jpg]] -->
![alt text](/MLP/Page45.jpg)
<!-- ![[Deep Learning IA306/Chapter 1/Page45.jpg]] -->
Neuron = projection, one hidden layer with 4 neurons = 4 projections
![alt text](/MLP/Page46.jpg)
<!-- ![[Deep Learning IA306/Chapter 1/Page46.jpg]] -->
![alt text](/MLP/Page47.jpg)
<!-- ![[Deep Learning IA306/Chapter 1/Page47.jpg]] -->
![alt text](/MLP/Page48.jpg)
<!-- ![[Deep Learning IA306/Chapter 1/Page48.jpg]] -->
![alt text](/MLP/Page49.jpg)
<!-- ![[Deep Learning IA306/Chapter 1/Page49.jpg]] -->
![alt text](/MLP/Page50.jpg)
<!-- ![[Deep Learning IA306/Chapter 1/Page50.jpg]] -->
![alt text](/MLP/Page51.jpg)
<!-- ![[Deep Learning IA306/Chapter 1/Page51.jpg]] -->
![alt text](/MLP/Page52.jpg)
<!-- ![[Deep Learning IA306/Chapter 1/Page52.jpg]] -->
![alt text](/MLP/Page53.jpg)
<!-- ![[Deep Learning IA306/Chapter 1/Page53.jpg]] -->

back-propagation = derivative of sigmoid function  -> Vanishing gradient ?
	
	The vanishing gradient problem occurs when the derivatives of the activation functions become very small as the network becomes deeper. This can happen when the sigmoid function is used as the activation function in the hidden layers of the network. The small derivatives make it difficult for the backpropagation algorithm to update the weights and biases of the network, which can prevent the network from learning.


![alt text](/MLP/Page55.jpg)
<!-- ![[Deep Learning IA306/Chapter 1/Page55.jpg]] -->
![alt text](/MLP/Page56.jpg)
<!-- ![[Deep Learning IA306/Chapter 1/Page56.jpg]] -->
![alt text](/MLP/Page57.jpg)
<!-- ![[Deep Learning IA306/Chapter 1/Page57.jpg]] -->
![alt text](/MLP/Page58.jpg)
<!-- ![[Deep Learning IA306/Chapter 1/Page58.jpg]] -->

Input from the storage = the results that are already computed.

![alt text](/MLP/Page59.jpg)

<!-- ![[Deep Learning IA306/Chapter 1/Page59.jpg]] -->


![alt text](/MLP/Page59.jpg)
<!-- ![[Deep Learning IA306/Chapter 1/Page59.jpg]] -->
We see that in case of 2 layers neural network, if we use linear activation function, the network reduces to a simple linear function. The linear activation function is only interesting for the last layer $g^{[L]}$ of regression problem.
![alt text](/MLP/Page62.jpg)

<!-- ![[Deep Learning IA306/Chapter 1/Page62.jpg]] -->
![alt text](/MLP/Page63.jpg)
<!-- ![[Deep Learning IA306/Chapter 1/Page63.jpg]] -->
Problem with **sigmoid** and **tanh**:
- if $z$ is very small (negative) or very large (positive) -> Slope becomes zero -> slow down gradient descent.
![alt text](/MLP/Page64.jpg)
<!-- ![[Deep Learning IA306/Chapter 1/Page64.jpg]] -->
![alt text](/MLP/Page65.jpg)
<!-- ![[Deep Learning IA306/Chapter 1/Page65.jpg]] -->
RELU has advantage and disadvantage:
- Advantages: doesn't vanish the gradient, prone the neural network
- Disvantages, it will kill some neurons, when $a=0$, that neuron is dead. 
![alt text](/MLP/Page66.jpg)
<!-- ![[Deep Learning IA306/Chapter 1/Page66.jpg]] -->
![alt text](/MLP/Page67.jpg)
<!-- ![[Deep Learning IA306/Chapter 1/Page67.jpg]] -->


### Regression
---
![alt text](/MLP/Page70.jpg)
<!-- ![[Deep Learning IA306/Chapter 1/Page70.jpg]] -->
We assume that our data follow the normal distribution with noise $\sigma$. 
![alt text](/MLP/Page71.jpg)
<!-- ![[Deep Learning IA306/Chapter 1/Page71.jpg]] -->
![alt text](/MLP/Page73.jpg)
<!-- ![[Deep Learning IA306/Chapter 1/Page73.jpg]] -->
![alt text](/MLP/Page74.jpg)
<!-- ![[Deep Learning IA306/Chapter 1/Page74.jpg]] -->
![alt text](/MLP/Page75.jpg)
<!-- ![[Deep Learning IA306/Chapter 1/Page75.jpg]] -->


![alt text](/MLP/Page77.jpg)
<!-- ![[Deep Learning IA306/Chapter 1/Page77.jpg]] -->
![alt text](/MLP/Page78.jpg)
<!-- ![[Deep Learning IA306/Chapter 1/Page78.jpg]] -->
In the multi-class classification or regression, we follow the same pattern as single class, but we have several output, and we try to minimise the sum of the loss. 
- BCE in case of multi-label classification -> several label possible
- Categorical cross-entropy in case of multi-class classification -> one possible label
Sigmoid will use when we have binary classification or multi label classification, when we have a mutli-class classification, we will use softmax.
![alt text](/MLP/Page79.jpg)
<!-- ![[Deep Learning IA306/Chapter 1/Page79.jpg]] -->
![alt text](/MLP/Page80.jpg)
<!-- ![[Deep Learning IA306/Chapter 1/Page80.jpg]] -->
![alt text](/MLP/Page81.jpg)
<!-- ![[Deep Learning IA306/Chapter 1/Page81.jpg]] -->
![alt text](/MLP/Page83.jpg)
<!-- ![[Deep Learning IA306/Chapter 1/Page83.jpg]] -->
### Gradient Descent 
---
**Batch GD:**
- Advantages:
    - Converges to the global minimum of the loss function, if it exists.
    - More stable than SGD.
- Disadvantages:
    - Can be computationally expensive for large training datasets.
    - Can be sensitive to the initial values of the model parameters.
**Stochastic GD:**
- Advantages:
    - Fast to converge, especially for large training datasets.
    - Less sensitive to the initial values of the model parameters.
- Disadvantages:
    - Can converge to a local minimum of the loss function, rather than the global minimum.
    - More noisy than BGD.
**Mini-batch GD:**
- Advantages:
    - Combines the advantages of BGD and SGD.
    - Faster than BGD for large training datasets.
    - More stable than SGD.
- Disadvantages:
    - Can converge to a local minimum of the loss function, rather than the global minimum.
![alt text](/MLP/Page85.jpg)
<!-- ![[Deep Learning IA306/Chapter 1/Page85.jpg]] -->
![alt text](/MLP/Page86.jpg)
<!-- ![[Deep Learning IA306/Chapter 1/Page86.jpg]] -->
Momentum GD, or gradient descent with momentum, is an extension to the gradient descent algorithm that can help to accelerate the convergence to the minimum of the loss function. It works by adding a momentum term to the gradient update equation. The momentum term is a weighted average of the previous gradients, which helps to push the update in the direction of the overall trend.w
![alt text](/MLP/Page87.jpg)
<!-- ![[Deep Learning IA306/Chapter 1/Page87.jpg]] -->
![alt text](/MLP/Page88.jpg)
<!-- ![[Deep Learning IA306/Chapter 1/Page88.jpg]] -->
![alt text](/MLP/Page89.jpg)
<!-- ![[Deep Learning IA306/Chapter 1/Page89.jpg]] -->

Adaptive learning rate algorithms are a class of optimization algorithms that automatically adjust the learning rate during training. This can help to improve the performance of the training process by ensuring that the learning rate is always appropriate for the current state of the model.
![alt text](/MLP/Page90.jpg)
<!-- ![[Deep Learning IA306/Chapter 1/Page90.jpg]] -->
![alt text](/MLP/Page91.jpg)
<!-- ![[Deep Learning IA306/Chapter 1/Page91.jpg]] -->
![alt text](/MLP/Page92.jpg)
<!-- ![[Deep Learning IA306/Chapter 1/Page92.jpg]] -->
**Adam** is an algorithm that combines the advantages of momentum GD and AdaGrad. It uses a momentum term to accelerate the convergence to the minimum of the loss function, and it uses an adaptive learning rate to ensure that the learning rate is always appropriate for the current state of the model.
![alt text](/MLP/Page93.jpg)
<!-- ![[Deep Learning IA306/Chapter 1/Page93.jpg]] -->
![alt text](/MLP/Page94.jpg)
<!-- ![[Deep Learning IA306/Chapter 1/Page94.jpg]] -->
![alt text](/MLP/Page95.jpg)
<!-- ![[Deep Learning IA306/Chapter 1/Page95.jpg]] -->
![alt text](/MLP/Page96.jpg)
<!-- ![[Deep Learning IA306/Chapter 1/Page96.jpg]] -->
![alt text](/MLP/Page97.jpg)
<!-- ![[Deep Learning IA306/Chapter 1/Page97.jpg]] -->
![alt text](/MLP/Page98.jpg)
<!-- ![[Deep Learning IA306/Chapter 1/Page98.jpg]] -->

### Weight Initialisation
---
![alt text](/MLP/Page100.jpg)
<!-- ![[Deep Learning IA306/Chapter 1/Page100.jpg]] -->
![alt text](/MLP/Page101.jpg)
<!-- ![[Deep Learning IA306/Chapter 1/Page101.jpg]] -->
![alt text](/MLP/Page102.jpg)
<!-- ![[Deep Learning IA306/Chapter 1/Page102.jpg]] -->
![alt text](/MLP/Page103.jpg)
<!-- ![[Deep Learning IA306/Chapter 1/Page103.jpg]] -->
There are a few reasons why it is important to initialize the weights of a neural network to be around 0.

- **To avoid symmetry:** If all of the weights in a neural network are initialized to the same value, then the network will be symmetric. This means that all of the neurons will learn the same things, and the network will not be able to learn complex relationships between the input and output variables.
- **To avoid vanishing gradients:** If the weights in a neural network are initialized to be too large, then the gradients of the loss function may become very small. This can make it difficult or even impossible for the network to learn.
- **To improve the performance of the network:** Studies have shown that initializing the weights of a neural network to be around 0 can improve the performance of the network on a variety of tasks.
![alt text](/MLP/Page104.jpg)
<!-- ![[Deep Learning IA306/Chapter 1/Page104.jpg]] -->

### Regularisation
---
![alt text](/MLP/Page106.jpg)
<!-- ![[Deep Learning IA306/Chapter 1/Page106.jpg]] -->
- **L1 regularization**, also known as Lasso regularization, adds a penalty term to the loss function that is proportional to the absolute value of the model parameters. This penalty term encourages the model to select a smaller subset of features that are important for making predictions. In other words, L1 regularization can be used for feature selection.
	- **The absolute value penalty function in L1 regularization encourages the model to shrink the weights towards zero. This is because the penalty is proportional to the absolute value of the weight, so the model can minimize the penalty by making the weight zero. When the model shrinks the weights towards zero, some of the weights will eventually become zero. This results in a sparse weight vector, where many of the weights are zero.**
- **L2 regularization**, also known as Ridge regularization, adds a penalty term to the loss function that is proportional to the square of the model parameters. This penalty term encourages the model to distribute the weights evenly across all of the features. By doing so, L2 regularization can prevent the model from becoming too dependent on any one feature.
	- **The square penalty function in L2 regularization encourages the model to shrink the weights towards zero, but not as much as L1 regularization. This is because the penalty is proportional to the square of the weight, so the model can minimize the penalty by making the weights small, but not necessarily zero. When the model shrinks the weights towards zero, the magnitude of all of the weights will be reduced. However, the weights will not necessarily become zero. This results in a weight vector where all of the weights are small, but not necessarily zero.**
- Both L1 and L2 regularization can help to reduce model complexity by shrinking the model parameters towards zero. This makes the model less sensitive to noise in the training data and can improve the generalization performance of the model.
- Elastic net regularization combines the advantages of L1 and L2 regularization. It can produce sparse weight vectors, like L1 regularization, and it can prevent overfitting, like L2 regularization.
- **Heavy weights can imply overfitting. This is because heavy weights indicate that the model is relying heavily on a small number of features to make predictions. If these features are not representative of the data that the model will be used on in production, then the model is likely to overfit.**
![alt text](/MLP/Page107.jpg)
<!-- ![[Deep Learning IA306/Chapter 1/Page107.jpg]] -->
![alt text](/MLP/Page108.jpg)
<!-- ![[Deep Learning IA306/Chapter 1/Page108.jpg]] -->
![alt text](/MLP/Page109.jpg)
<!-- ![[Deep Learning IA306/Chapter 1/Page109.jpg]] -->

Dropout regularization is a technique that can be used to reduce the risk of overfitting in machine learning models. It works by randomly dropping out neurons from the network during training. This forces the model to learn to rely on a wider range of features, which can help to improve the generalization performance of the model.
![alt text](/MLP/Page110.jpg)
<!-- ![[Deep Learning IA306/Chapter 1/Page110.jpg]] -->

### Normalisation
---
![alt text](/MLP/Page112.jpg)
<!-- ![[Deep Learning IA306/Chapter 1/Page112.jpg]] -->
Normalization scales the data so that it falls within a certain range, such as 0 to 1 or -1 to 1. This is done by subtracting the minimum value from the data and then dividing by the range of the data.

The following equation shows how to normalize data:
```
normalized_data = (data - min(data)) / (max(data) - min(data))
```
Standardization scales the data so that it has a mean of 0 and a standard deviation of 1. This is done by subtracting the mean from the data and then dividing by the standard deviation of the data.

The following equation shows how to standardize data:

```
standardized_data = (data - mean(data)) / std(data)
```
![alt text](/MLP/Page113.jpg)
<!-- ![[Deep Learning IA306/Chapter 1/Page113.jpg]] -->

**Batch normalization (BN)** is a technique used to normalize the activations of each layer of a neural network during training. It does this by subtracting the mean and dividing by the standard deviation of the activations across the batch. This helps to stabilize the training process and improve the performance of the model.
![alt text](/MLP/Page114.jpg)
<!-- ![[Deep Learning IA306/Chapter 1/Page114.jpg]] -->
![alt text](/MLP/Page115.jpg)
<!-- ![[Deep Learning IA306/Chapter 1/Page115.jpg]] -->
![alt text](/MLP/Page116.jpg)
<!-- ![[Deep Learning IA306/Chapter 1/Page116.jpg]] -->
![alt text](/MLP/Page117.jpg)
<!-- ![[Deep Learning IA306/Chapter 1/Page117.jpg]] -->
![alt text](/MLP/Page118.jpg)
<!-- ![[Deep Learning IA306/Chapter 1/Page118.jpg]] -->