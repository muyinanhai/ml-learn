#Logistic Regression
---
Logistic regression sometimes called the logistic model or logit model, analyzes the relationship between multiple independent variables and a categorical dependent variable and estimates the probability of occurence of an event by fitting data to a logistic curve. There are two models of an logistic regression, binary logistic regression and multinominal logistic regression. Binary logistic regression is typically used when the dependent variable is dischotomous and the independent variables are either continuous or categorial. When the dependent varible is not dichotomous and is comprised of more than two categories, a multinomial logistic regression can be employed.[1]

##Definition
An explanation of logistic regression can begin with an explanation of the standard logistic function. The logistic function is useful because it can take an input with any value from negative to positive infinity, whereas the output always takes values between zero and one and hence is interpretable as a probability. The logistic function  $\sigma(t)$ is defined as follows:[2]
$$
\sigma(t)=\frac{e^t}{1+e^t}=\frac{1}{1+e^{-t}}
$$
A graph of the logistic function on the t-interval (-6,6) is shown as below:
<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/8/88/Logistic-curve.svg/600px-Logistic-curve.svg.png"
/>

In linear regresion
$$
h_\theta(x)=\theta^Tx
$$
When using logistic regresion, we want the output values between 0 and 1,so
$$
h_\theta(x)=\sigma(\theta^Tx)=\frac{1}{1+e^{-\theta^Tx}}
$$

so we can define the cost function of logistic regression:
$$
cost(h_\theta(x),y)=
\left\{\begin{matrix}
h_\theta(x)...............y=1
\\ 1-h_\theta(x)...........y=0
\end{matrix}\right.
$$
we can simplify the cost function as:
$$
cost(h_\theta(x),y)=y*h_\theta(x)+(1-y)*(1-h_\theta(x))
$$

then the objective function of logistic regression can be defined as below:
$$
maxmize \prod_{i=1}^{m}cost(h_\theta(x),y)=\prod_{i=1}^{m}\{y*h_\theta(x)+(1-y)*(1-h_\theta(x))\}
$$
**Equivalent to**
$$
J(\theta)=\mathbf{maxmize}\{-\frac{1}{m}\sum_{i=1}^{m}log(y*h_\theta(x)+(1-y)*(1-h_\theta(x)))\}
$$
since when y=1 
$$
log(y*h_\theta(x)+(1-y)*(1-h_\theta(x)))=log(h_\theta(x))
$$
when y=0
$$
log(y*h_\theta(x)+(1-y)*(1-h_\theta(x)))=log(1-h_\theta(x))
$$
so
$$
J(\theta)=\mathbf{maxmize}\{-\frac{1}{m}\sum_{i=1}^{m}y*log(h_\theta(x))+(1-y)*log(1-h_\theta(x))\}
$$

Since
$$
\sigma(t)^{'}=\sigma(t)(1-\sigma(t))
$$
$$
(1-\sigma(t))^{'}=\sigma(t)(\sigma(t)-1)
$$
so
$$
\frac{\delta J(\theta)}{\delta\theta_j}=-\frac{1}{m}\sum_{i=1}^{m}{y*\frac{1}{h_\theta(x)}*{h_\theta(x)}^{'}+(1-y)*\frac{1}{1-h_\theta(x)}*(1-h_\theta(x))^{'}}
$$
then
$$
\frac{\delta J(\theta)}{\delta\theta_j}=-\frac{1}{m}\sum_{i=1}^{m}{y*\frac{1}{h_\theta(x)}*h_\theta(x)*{(1-h_\theta(x))*x_j}+(1-y)*\frac{1}{1-h_\theta(x)}*h_\theta(x)(h_\theta(x)-1)*x_j}
$$
then
$$
\frac{\delta J(\theta)}{\delta\theta_j}=-\frac{1}{m}\sum_{i=1}^{m}{y*{(1-h_\theta(x))*x_j}-(1-y)*h_\theta(x)*x_j}
$$
then
$$
\frac{\delta J(\theta)}{\delta\theta_j}=\frac{1}{m}\sum_{i=1}^{m}{(h_\theta(x)-y)*x_j}
$$


[1]http://synapse.koreamed.org/Synapse/Data/PDFData/0006JKAN/jkan-43-154.pdf
[2]https://en.wikipedia.org/wiki/Logistic_regression

