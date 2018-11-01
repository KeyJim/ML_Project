
# 1 Introduction to neural network 

## Perception --- artificial neuron


#### define
<div align="center">  <img src="https://github.com/LiuChuang0059/ML_Project/blob/master/Picture/perception.png" width="400"/> </div><br>


* 给对应值一个权重以及一个 阈值   ； 大于阈值表示为1，小于阈值表示为0.

* a many-layer network of perceptrons can engage in sophisticated decision making.

<div align="center">  <img src="https://github.com/LiuChuang0059/ML_Project/blob/master/Picture/perception_complax.png" width="400"/> </div><br>
**类似于量子力学里面的一维无限深势井 + 迪立科立函数**

#### Apply 1 --- 预测（0 or 1）：
> Suppose the weekend is coming up, and you've heard that there's going to be a cheese festival in your city. You like cheese, and are trying to decide whether or not to go to the festival. You might make your decision by weighing up three factors:

> For instance, we'd have x1=1 if the weather is good, and x1=0 if the weather is bad. Similarly, x2=1 if your boyfriend or girlfriend wants to go, and x2=0 if not. And similarly again for x3 and public transit.

根据各个因素影响程度选定权重系数weight


#### Apply 2 ---计算逻辑函数 ： AND OR 之类


<div align="center">  <img src="https://github.com/LiuChuang0059/ML_Project/blob/master/Picture/NAND_gate.png" width="400"/> </div><br>

<div align="center">  <img src="https://github.com/LiuChuang0059/ML_Project/blob/master/Picture/NAND_adder.png" width="400"/> </div><br>

----------

## Sigmoid neurons

* similar with Perception

* modified so that small changes in their weights and bias cause only a small change in their output


--------
## The architecture of neural networks

* hidden layer --- it really means nothing more than "not an input or an output"

* feedforward neural network---where the output from one layer is used as input to the next layer



<div align="center">  <img src="https://github.com/LiuChuang0059/ML_Project/blob/master/Picture/NN_structure.png" width="400"/> </div><br>



# 2 A simple network to classify handwritten digits

* example： recognize handwritten digits

* pixels ：28x28 =784  ----- 784 neurons


* Goal: Using MLP（2 layers）to  implete complex func (784 ---> 10)

<div align="center">  <img src="https://github.com/LiuChuang0059/ML_Project/blob/master/Picture/MLP33.09.png" width="400"/> </div><br>

* First recognise little pattern，Then  Bigger pattern
<div align="center">  <img src="https://github.com/LiuChuang0059/ML_Project/blob/master/Picture/MLP24.14.png" width="400"/> </div><br>


* Function mapped between layers
<div align="center">  <img src="https://github.com/LiuChuang0059/ML_Project/blob/master/Picture/MLP35.16.png" width="400"/> </div><br>


<div align="center">  <img src="https://github.com/LiuChuang0059/ML_Project/blob/master/Picture/MLP31.55.png" width="400"/> </div><br>

<div align="center">  <img src="https://github.com/LiuChuang0059/ML_Project/blob/master/Picture/MLP31.14.png" width="400"/> </div><br>


* The number of parameters that need to be determined for the complex function of the entire neuron
<div align="center">  <img src="https://github.com/LiuChuang0059/ML_Project/blob/master/Picture/MLP29.21.png" width="500"/> </div><br>

-----

* To configure all parameters manually, introducing a gradient descent algorithm。

  * Random Initialise
  
  * Cost func
  <div align="center">  <img src="https://github.com/LiuChuang0059/ML_Project/blob/master/Picture/MLP37.15.png" width="500"/> </div><br>
  
  
  * Change weight and bias 
  
* Overall identification process

<div align="center">  <img src="https://github.com/LiuChuang0059/ML_Project/blob/master/Picture/MLP47.13.png" width="400"/> </div><br>



<div align="center">  <img src="https://github.com/LiuChuang0059/ML_Project/blob/master/Picture/MLP47.36.png" width="400"/> </div><br>  
  
 
<div align="center">  <img src="https://github.com/LiuChuang0059/ML_Project/blob/master/Picture/MLP45.27.png" width="400"/> </div><br>  



----


**BP Alg **----Calculate the gradient of the previous step


* Goal : Know the direction in which the activation value should change, know the size of the activation value change


<div align="center">  <img src="https://github.com/LiuChuang0059/ML_Project/blob/master/Picture/BP40.00.png" width="400"/> </div><br>

* How to change ---3 parameters


<div align="center">  <img src="https://github.com/LiuChuang0059/ML_Project/blob/master/Picture/BP44.47.png" width="400"/> </div><br>

* 不但是对于目标数字的变化（变为1），还应该是所有其他的（变为0） 求和平均

<div align="center">  <img src="https://github.com/LiuChuang0059/ML_Project/raw/master/Picture/BP47.33.png" width="400"/> </div><br>


* Perform bp on all data in the data set to average the parameters to determine the final value.



* But all the data is requested every time, the calculation speed is too slow, 
* We can use the random gradient descent method - randomly divided into minibatch, calculate the gradient of each minibatch until all calculations are completed.
 
 ------------
 
 
**The mathematical expression of the BP algorithm**----mainly understood by the chain law

<div align="center">  <img src="https://github.com/LiuChuang0059/ML_Project/blob/master/Picture/BP41.19.png" width="400"/> </div><br>


<div align="center">  <img src="https://github.com/LiuChuang0059/ML_Project/blob/master/Picture/BP58.55.png" width="400"/> </div><br>

* Note that the differential of the previous activation value has a summation because it will contribute to the activation value of each next layer.
<div align="center">  <img src="https://github.com/LiuChuang0059/ML_Project/blob/master/Picture/BP38.59.png" width="400"/> </div><br>


-------
------

# 3   --- A neural network can compute any function

### Theorems

  
  * we can get an approximation .By increasing the number of hidden neurons, we can improve the approximation. We can set a desired accuracy.
  
  * continuous functions better .

<div align="center">  <img src="https://github.com/LiuChuang0059/ML_Project/blob/master/Picture/Netural-network.png" width="400"/> </div><br>

*  even function has many inputs,  and many outputs

*  even a  single hidden layer


### 1 : One input and one output

* Single hidden layer : two neurons.  σ(wx+b), where σ(z)≡1/(1+e−z)

* Focus on the top neuron


* Since the w is large enough ,we can get the Step func 
<div align="center">  <img src="https://github.com/LiuChuang0059/ML_Project/blob/master/Picture/Step-func.jpg" width="400"/> </div><br>

* We can find  the step is at position s=−b/w
<div align="center">  <img src="https://github.com/LiuChuang0059/ML_Project/blob/master/Picture/step_position.png" width="400"/> </div><br>

* Focus on entire network

<div align="center">  <img src="https://github.com/LiuChuang0059/ML_Project/blob/master/Picture/Entire%20network.png" width="400"/> </div><br>

* We can get a bump function ---  set the height :h
<div align="center">  <img src="https://github.com/LiuChuang0059/ML_Project/blob/master/Picture/bump_function.jpg" width="400"/> </div><br>

* We can use our bump-making trick to get two bumps, by gluing two pairs of hidden neurons together into the same network:

* By changing the output weights we're actually designing the function

<div align="center">  <img src="https://github.com/LiuChuang0059/ML_Project/blob/master/Picture/5bump_func.png" width="400"/> </div><br>

<div align="center">  <img src="https://github.com/LiuChuang0059/ML_Project/blob/master/Picture/bump-func2.png" width="400"/> </div><br>

### 2 Many input values

* With w2=0 the input y makes no difference to the output from the neuron

*  The actual location of the step point is sx≡−b/w1.

<div align="center">  <img src="https://github.com/LiuChuang0059/ML_Project/blob/master/Picture/2D-X.png" width="400"/> </div><br>

* we can get from X+Y
<div align="center">  <img src="https://github.com/LiuChuang0059/ML_Project/blob/master/Picture/2D-XY.png" width="400"/> </div><br>

----------------
----------------

# 4  ---  Improving the way neural networks learn


### 1. The cross-entropy cost function

> Artificial neuron has a lot of difficulty learning when it's badly wrong - far more difficulty than when it's just a little wrong

* neuron's output is close to 1, the curve gets very flat, ∂C/∂w and ∂C/∂b get very small. 

<div align="center">  <img src="https://github.com/LiuChuang0059/ML_Project/blob/master/Picture/Sigmoid-func.png" width="400"/> </div><br>

*  cross-entropy cost function

<div align="center">  <img src="https://github.com/LiuChuang0059/ML_Project/blob/master/Picture/Cross.png" width="400"/> </div><br>
  
  * the cross-entropy is positive
  
  * tends toward zero as the neuron gets better at computing the desired output
  
  * it avoids the problem of learning slowing down: rate at which the weight learns is controlled by σ(z)−y ---- the error in the output


<div align="center">  <img src="https://github.com/LiuChuang0059/ML_Project/blob/master/Picture/Decent.png" width="400"/> </div><br>


### 2 Softmax
* If one activation increases, then the other output activations must decrease by the same total amount, to ensure the sum over all activations remains 1

* the output from the softmax layer can be thought of as a probability distribution.



<div align="center">  <img src="https://github.com/LiuChuang0059/ML_Project/blob/master/Picture/SoftMax.png" width="400"/> </div><br>


* log-likelihood cost function to solve learning slowdown problem



### 3 Overfitting and regularization   ----- Small

#### 1 Overfit 

* To detect overfitting : we can keep  track of accuracy on the test data as our network trains. 

* Better to test data and the training data both stop improving at the same time.

------
**Early Stopping**

* use the validation_data (different from train_data test_data) to prevenr overfitting

  > Compute the classification accuracy on the validation_data at the end of each epoch. Once the classification accuracy on the validation_data has saturated, we stop trainin

* Why not test_data: we may end up finding hyper-parameters which fit particular peculiarities of the test_data, but where the performance of the network won't generalize to other data sets

* Once we've got the hyper-parameters we want, we do a final evaluation of accuracy using the test_data


#### 2 Regularization---- reduce overfitting and to increase classification accuracies(weight decay)
> The idea of L2 regularization is to add an extra term to the cost function, a term called the regularization term.

<div align="center">  <img src="https://github.com/LiuChuang0059/ML_Project/blob/master/Picture/Regulation.png" width="400"/> </div><br>

* add the sum of the squares of all the weights in the network

  *  a way of compromising between finding small weights and minimizing the original cost function
  
  * when λ is small we prefer to minimize the original cost function, but when λ is large we prefer small weights.

------
**Why?**
> smaller weights are, in some sense, lower complexity, and so provide a simpler and more powerful explanation for the data, and should thus be preferred

* higher order really just learning the effects of local noise

*  The smallness of the weights means that the behaviour of the network won't change too much if we change a few random inputs here and there. That makes it difficult for a regularized network to learn the effects of local noise in the data

> sometimes the more complex explanation turns out to be correct.

#### 3 Other techniques for regularization

* L1 regularization
  * In L1 regularization, the weights shrink by a constant amount toward 0. 
  * In L2 regularization, the weights shrink by an amount which is proportional to 

---------

* Dropout--- modify the network itself
  > randomly and temporarily deleting half the hidden neurons in the network,while leaving the input and output neurons untouched

*  forward-propagate the input x through the modified network
*  update the appropriate weights and biases
*  restoring the dropout neurons, choosing a new random subset of hidden neurons to delete
*  we actually run the full network , halve the weights outgoing from the hidden neurons.

-----------

**Why**

* The different networks may overfit in different ways, and averaging may help eliminate that kind of overfitting.

* The dropout procedure is like averaging the effects of a very large number of different networks.

* if we think of our network as a model which is making predictions, then we can think of dropout as a way of making sure that the model is robust to the loss of any individual piece of evidence

  > "This technique reduces complex co-adaptations of neurons, since a neuron cannot rely on the presence of particular other neurons. It is, therefore, forced to learn more robust features that are useful in conjunction with many different random subsets of the other neurons." ---- *ImageNet Classification with Deep Convolutional Neural Networks, by Alex Krizhevsky, Ilya Sutskever, and Geoffrey Hinton (2012).


**A combination of dropout and a modified form of L2 regularization**


#### 4 Artificially expanding the training data

-------

### 4 Choose a neural network's hyper-parameters


* speed up experimentation 
  *  The simplest network likely to do meaningful learning
  *  Increasing the frequency of monitoring

-----

* Learning rate:
  * If η is too large then the steps will be so large that they may actually overshoot the minimum
  
  * Choosing η so small slows down stochastic gradient descent

      * First, we estimate the threshold value for η  
      
      * A factor of two below the threshold
      
      * Its primary purpose is really to control the step size in gradient descent, and monitoring the training cost is the best way to detect if the step size is too big.
  
  **vary the learning rate**
  
  * To hold the learning rate constant until the validation accuracy starts to get worse, then decrease the learning rate by some amount, say a factor of two or ten.
  
 
 --------
 
* Training epochs: early stopping

  * At the end of each epoch , compute the classification accuracy on the validation data. When that stops improving **for quite some time**,terminate.
  > Using the no-improvement-in-ten rule for initial experimentation, and gradually adopting more lenient rule.

-------

* The regularization parameter, λ:
  * Starting initially with no regularization (λ=0.0)
  * Determining a value for η
  * Use the validation data to select a good value for λ
  * Start by trialling λ=1.0 , increase or decrease by factors of 10, depend on the validation data
  * Found a good order of magnitude, you can fine tune  λ.
  * re-optimize η again.
  
  ------
  
* Mini-batch size:
  * Too small, and you don't get to take full advantage of the benefits of good matrix libraries optimized for fast hardware. 
  
  * Too small, and you don't get to take full advantage of the benefits of good matrix libraries optimized for fast hardware. 
  
  
### 5 Automated techniques : 
  * grid search
    >  systematically searches through a grid in hyper-parameter space.
    > Random search for hyper-parameter optimization, by James Bergstra and Yoshua Bengio (2012).
  * Bayesian approach
  > Practical Bayesian optimization of machine learning algorithms, by Jasper Snoek, Hugo Larochelle, and Ryan Adams.



### 6 Other models of artificial neuron

*  tanh --- [-1 ,1]
*  max


### 7 Variations on stochastic gradient descent

* Hessian technique

* Momentum-based gradient descent





