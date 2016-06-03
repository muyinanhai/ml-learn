## Ridge Regression

The objective of ridge regression is to minimise the cost fucntion:
$$
J(\theta)=\frac{1}{2}\sum_{i=1}^{m}(h_{\theta}(x^{(i)})-y^{(i)})^{2}+\lambda\sum_{j=1}^{n}\theta_j^2
$$
$$
h_{\theta}=\theta^Tx=\theta_0+\theta_1x_1+...\theta_nx_n
$$

The parameters of your model are the $\theta$ and $\lambda$  values.
There are two kinds of functions to solve this problem:
####1:Least Square method
$$
\theta=(X^TX+\lambda I)^{-1}X^Ty
$$
####2:Batch Gradient Descent

Derivative of $J(\theta)$ on $\theta_j$:
$$
\frac{J(\theta)}{\theta_j}^{'}=\sum_{i=1}^{m}(h_{\theta}(x^{(i)})-y^{(i)})x_j+2\lambda\theta_j=(h_\theta(x)-y)x_j+2\lambda\theta_j
$$
