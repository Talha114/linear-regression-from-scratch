# linear-regression-from-scratch

Regression

Regression is a method of modeling a target value based on
independent predictors. This method is mostly used for forecasting and
finding out cause and effect relationship between variables. Regression
techniques mostly differ based on the number of independent variables and
the type of relationship between the independent and dependent variables.

Linear Regression

Simple linear regression is a type of regression analysis where the number of
independent variables is one and there is a linear relationship between the
independent(x) and dependent(y) variable. The line can be modeled based
on the linear equation shown below.

y = theta0 + theta1 * x ## Linear Equation

or

y = c + m* x ## Linear Equation

The motive of the linear regression algorithm is to find the best values for m
and c.( or theta0 and theta1) .
Before moving on to the algorithm, let’s have a look at two important concepts
you must know to better understand linear regression.

1-Cost Function ( or Loss Function)
The cost function helps us to figure out the best possible values for theta0
and theta1 which would provide the best fit line for the data points. Since we
want the best values for theta0 and theta1, we convert this search problem
into a minimization problem where we would like to minimize the error
between the predicted value and the actual value

We choose the above function to minimize. The difference between the
predicted values and ground truth measures the error difference. We square
the error difference and sum over all data points and divide that value by the
total number of data points. This provides the average squared error over all
the data points. Therefore, this cost function is also known as the Mean
Squared Error(MSE) function. Now, using this MSE function we are going to
change the values of a_0 and a_1 such that the MSE value settles at the
minima.
2-Optimizer
There are many optimizer in family of this type of algorithms but one of the
most widely used optimizer is Gradient Descent
Gradient Descent
The next important concept needed to understand linear regression is
gradient
descent. The idea is that we start with some values for theta0 and theta1 and
then we change these values iteratively to reduce the cost. Gradient descent
helps us on how to change the values.

To draw an analogy, imagine a pit in the shape of U and you are standing at
the topmost point in the pit and your objective is to reach the bottom of the pit.
There is a catch, you can only take a discrete number of steps to reach the
bottom. If you decide to take one step at a time you would eventually reach
the bottom of the pit but this would take a longer time. If you choose to take

longer steps each time, you would reach sooner but, there is a chance that
you could overshoot the bottom of the pit and not exactly at the bottom. In the
gradient descent algorithm, the number of steps you take is the learning rate.
This decides on how fast the algorithm converges to the minima

You may be wondering how to use gradient descent to update theta0 and
theta1. To update theta0 and theta_1, we take gradients from the cost
function. To find these gradients, we take partial derivatives with respect
to theta0 and
theta1.

Algorithm:
1) Pick initial values of thetas randomly
2) Calculate Prediction, Using hypothesis of line equation
3) Find Loss on your prediction
4) Find Rate of change of error (Using Derivative)
5) Update Thetas with following equation

newThetas = OldThetas – alpa*(Derivatives of thetas)

Termination Criteria
1) At certain amount of iterations
2) When value of cost function between 2 iterations are less than certain
threshold like 0.01 etc.

Lab Task

You need to implement Linear regression given the randomly generated
data in python notebook.
1) def hyp(theta, X): Implement hypothesis for equation of line.

2) def cost_function(theta,X,Y): Implement cost function ( Mean Squared
Error )
for linear regression training

3) def
derivative_cost_function(theta,X,Y): Implement Derivative of cost function
(error function ) to find rate of change of
error in regression

4) def
GradientDescent(X,Y,cost_function,derivative_cost_function,maxniter):
Implement gradient descent algorithm to train linear regression Model given
the following algorithm
for i in range(0, numiter):
# hyp=hypothesis=(theta,X)
# loss= hyp.T-Y
# Cost = sum(loss**2)/2.0*nexamples
# print cost
# gradiants= loss.T . X.T / nexamples

theta = theta - alpha * gradient
return theta
